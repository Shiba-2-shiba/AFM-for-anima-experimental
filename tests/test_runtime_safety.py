import json
from unittest.mock import patch
import pytest
import torch
import anima_afm as afm


def inputs(batch=2, heads=4):
    return (torch.randn(batch, heads, 16, 4), torch.randn(batch, heads, 5, 4), torch.randn(batch, heads, 5, 4), heads)


def options():
    return dict(skip_reshape=True, transformer_options=dict(sigmas=torch.tensor([1.]), sample_sigmas=torch.tensor([1., .5, 0.]), cond_or_uncond=[1, 0], afm_spatial_shape=[4, 4]))


def original(q,k,v,heads,**kwargs):
    return (torch.softmax(q @ k.transpose(-2,-1) / 2, -1) @ v).transpose(1,2).flatten(2)


def test_auto_does_not_guess_square_even_for_rectangular_square_product():
    assert afm.infer_spatial_shape(2304, afm.AFMConfig()).shape is None
    assert afm.infer_spatial_shape(2304, afm.AFMConfig(spatial_shape_mode='square_only')).shape == (48,48)


def test_auto_conflicting_sources_fail_closed():
    assert afm.infer_spatial_shape(16, afm.AFMConfig(), {'afm_spatial_shape':[4,4]}, {'afm_spatial_shape':[2,8]}).shape is None


def test_auto_invalid_strict_metadata_overrides_valid_candidate():
    assert afm.infer_spatial_shape(16, afm.AFMConfig(), {'afm_spatial_shape':[4,4], 'afm_latent_height':3,'afm_latent_width':4}).shape is None


@pytest.mark.parametrize('field', ['strength','beta','gamma','mask_width','max_peak_mib','max_logits_mib','aspect_tolerance'])
def test_config_rejects_nonfinite(field):
    with pytest.raises(ValueError):
        afm.AFMConfig(**{field:float('nan')}).validate()


def test_encoder_candidate_does_not_select_other_same_stage_block(tmp_path):
    path=tmp_path/'scope.json'
    path.write_text(json.dumps([dict(block_path='blocks.0', block_index=0, stage_tag='early', candidate=True)]))
    matcher=afm.ScopeMatcher(afm.AFMConfig(scope_mode='encoder_equivalent',scope_map_path=str(path)))
    identity=afm.ScopeIdentity('block.1','blocks.1','Block',1,'early','runtime')
    assert not matcher.select(identity).selected


def test_memory_estimate_covers_fft_live_tensors():
    est=afm.estimate_attention_memory(2,4,16,5)
    assert est.estimated_peak_mib >= 8*est.logits_mib


def test_original_exception_is_not_retried():
    calls=[]
    def broken(*args,**kwargs):
        calls.append(1)
        raise RuntimeError('original failed')
    with pytest.raises(RuntimeError,match='original failed'):
        afm.AnimaAFMAttentionOverride(afm.AFMConfig(mode='off'))(broken,*inputs(),**options())
    assert len(calls)==1


def test_observe_diagnostics_select_before_qk():
    override=afm.AnimaAFMAttentionOverride(afm.AFMConfig(mode='observe',spectral_diag='sampled', diagnostic_max_heads=1,diagnostic_max_batches=1))
    shapes=[]
    real=torch.matmul
    def record(a,b):
        shapes.append(tuple(a.shape))
        return real(a,b)
    with patch('anima_afm.torch.matmul',side_effect=record):
        override(original,*inputs(batch=4),**options())
    assert shapes and all(shape[:2] == (1,1) for shape in shapes)


def test_observe_guard_runs_before_qk():
    override=afm.AnimaAFMAttentionOverride(afm.AFMConfig(mode='observe',spectral_diag='full',max_peak_mib=0.00001))
    with patch('anima_afm.torch.matmul',side_effect=AssertionError('diagnostic allocation')):
        result=override(original,*inputs(),**options())
    assert result.shape==(2,16,16)
    assert override.stats.fallback_reasons['peak_vram_guard_exceeded']==1


def test_nonfinite_q_skips_fft_and_stops_nonfinite_original():
    args=list(inputs()); args[0][0,0,0,0]=float('nan')
    with patch('anima_afm.edit_logits_fft',side_effect=AssertionError('FFT must not run')):
        override=afm.AnimaAFMAttentionOverride(afm.AFMConfig())
        with pytest.raises(RuntimeError,match="nonfinite original"):
            override(original,*args,**options())
    assert override.stats.fallback_reasons['nonfinite_attention_input']==1


def test_model_evaluation_resets_call_selection_without_resetting_step_stats():
    override=afm.AnimaAFMAttentionOverride(afm.AFMConfig(target_call_indices='0'))
    for _ in range(2):
        override.begin_model_evaluation()
        override(original,*inputs(),**options())
    assert override.stats.edited_calls==2
    assert override.stats.steps[0].eligible_calls==2

@pytest.mark.parametrize('bad', [[0,0],[float('nan'),4],[-1,16]])
def test_invalid_explicit_runtime_shape_blocks_other_valid_metadata(bad):
    assert afm.infer_spatial_shape(16,afm.AFMConfig(),{'afm_spatial_shape':bad,'spatial_shape':[4,4]}).shape is None


def test_temporal_grid_flag_blocks_explicit_mode_too():
    config=afm.AFMConfig(spatial_shape_mode='explicit_latent',latent_height=4,latent_width=4)
    assert afm.infer_spatial_shape(16,config,{'afm_spatial_shape_invalid':True}).shape is None


def test_block_wrapper_does_not_guess_tensor_layout():
    assert afm._spatial_shape_candidates_from_call((torch.zeros(2,36,64,8),),{}) == []


def test_failed_diagnostic_does_not_claim_edit_was_returned():
    override=afm.AnimaAFMAttentionOverride(afm.AFMConfig(spectral_diag='sampled'))
    with patch.object(override,'_compute_edit_diagnostics',side_effect=RuntimeError('diagnostic failed')):
        override(original,*inputs(),**options())
    assert override.stats.edited_calls == 0


def test_encoder_candidate_does_not_expand_through_default_stage_scope(tmp_path):
    path=tmp_path/'scope.json'; path.write_text(json.dumps([dict(block_path='blocks.0',candidate=True)]))
    matcher=afm.ScopeMatcher(afm.AFMConfig(scope_mode='encoder_equivalent',scope_map_path=str(path),stage_scope='early'))
    assert not matcher.select(afm.ScopeIdentity('b1','blocks.1','Block',1,'early','runtime')).selected


def test_diagnostic_guard_estimate_is_reported_without_reducing_it():
    override=afm.AnimaAFMAttentionOverride(afm.AFMConfig(spectral_diag='sampled'))
    context=override._prepare_context(original, inputs(),options())
    estimate=override._guard_allocation(context,2,4,True)
    assert context.step.max_estimated_peak_mib == estimate['estimated_peak_mib']


def test_finite_inputs_overflow_diagnostic_logits_are_not_logged():
    args=list(inputs()); args[0].fill_(1e30);args[1].fill_(1e30)
    override=afm.AnimaAFMAttentionOverride(afm.AFMConfig(mode='observe',spectral_diag='sampled'))
    with pytest.raises(RuntimeError,match='nonfinite original'):
        override(original,*args,**options())
    assert override.stats.fallback_reasons['nonfinite_attention_logits']==1


def test_float64_backend_is_preserved_instead_of_downcast_manual_path():
    args=[t.double() if torch.is_tensor(t) else t for t in inputs()]
    override=afm.AnimaAFMAttentionOverride(afm.AFMConfig())
    result=override(original,*args,**options())
    assert result.dtype==torch.float64
    assert override.stats.fallback_reasons['unsupported_attention_dtype']==1


def test_nonfinite_scale_uses_original_once():
    calls=[]
    def backend(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    override=afm.AnimaAFMAttentionOverride(afm.AFMConfig())
    override(backend,*inputs(),scale=float('inf'),**options())
    assert calls==[1]
    assert override.stats.fallback_reasons['nonfinite_attention_scale']==1


def test_mask_is_never_discarded_by_manual_path():
    calls=[]
    def backend(*args,**kwargs):
        calls.append(kwargs['mask']);return original(*args,**kwargs)
    mask=torch.ones(16,5,dtype=torch.bool)
    override=afm.AnimaAFMAttentionOverride(afm.AFMConfig())
    override(backend,*inputs(),mask=mask,**options())
    assert calls[0] is mask
    assert override.stats.fallback_reasons['mask_shape_unsupported']==1


def test_diagnostic_failure_after_partial_original_only_calls_original_once():
    calls=[]
    def backend(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    override=afm.AnimaAFMAttentionOverride(afm.AFMConfig(branch_mode='positive_only',spectral_diag='sampled'))
    with patch.object(override,'_compute_edit_diagnostics',side_effect=RuntimeError('failed')):
        result=override(backend,*inputs(),**options())
    assert calls==[1]
    assert override.stats.edited_calls==0


def test_rectangular_fft_transpose_equivariance():
    logits=torch.randn(1,2,24,5)
    config=afm.AFMConfig()
    transposed=logits.reshape(1,2,4,6,5).transpose(2,3).reshape(1,2,24,5)
    expected=afm.edit_logits_fft(logits,(4,6),1.2,1.1,config).reshape(1,2,4,6,5).transpose(2,3).reshape(1,2,24,5)
    actual=afm.edit_logits_fft(transposed,(6,4),1.2,1.1,config)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)

@pytest.mark.parametrize('bad', ['bad',[1,2,3],{'height':0,'width':4}])
def test_malformed_runtime_shape_cannot_be_overridden_by_valid_shape(bad):
    assert afm.infer_spatial_shape(16,afm.AFMConfig(),{'afm_spatial_shape':bad,'spatial_shape':[4,4]}).shape is None


def test_log_failure_does_not_discard_successfully_computed_edit():
    args=inputs()
    override=afm.AnimaAFMAttentionOverride(afm.AFMConfig(strength=.5))
    expected=afm.AnimaAFMAttentionOverride(afm.AFMConfig(strength=.5))(original,*args,**options())
    with patch.object(override,'_log_eligible_from_context',side_effect=OSError('log unavailable')):
        actual=override(original,*args,**options())
    torch.testing.assert_close(actual,expected)
    assert override.stats.edited_calls==1
    assert override.stats.fallback_calls==0
    assert override.telemetry_errors==1


def test_nonfinite_original_fallback_stops_without_retry():
    calls=[]
    def backend(*args,**kwargs):
        calls.append(1);return torch.full((2,16,16),float('nan'))
    override=afm.AnimaAFMAttentionOverride(afm.AFMConfig(mode='observe'))
    with pytest.raises(RuntimeError,match='nonfinite original'):
        override(backend,*inputs(),**options())
    assert calls==[1]


def test_constant_concentration_metric_is_explicitly_undefined():
    logits=torch.zeros(1,2,16,10)
    diagnostic=afm.sampled_spectral_diagnostics(logits,logits,(4,4),afm.AFMConfig(spectral_diag='sampled'),torch.tensor([0]),1,[0])[0]
    assert diagnostic.metric_defined is False
    assert diagnostic.metric_undefined_reason=='zero_or_nonfinite_centered_energy'


def test_undefined_metric_is_excluded_from_aggregate_summary():
    override=afm.AnimaAFMAttentionOverride(afm.AFMConfig(mode='observe',spectral_diag='sampled'))
    args=list(inputs());args[0].zero_();args[1].zero_()
    override(original,*args,**options())
    summary=override._spectral_summary(override.stats.steps[0])
    assert summary['spectral_delta_rho_mean'] is None
    assert summary['spectral_undefined_count'] > 0
    assert summary['spectral_emitted_count'] == summary['spectral_diag_count'] + summary['spectral_undefined_count']


def test_sampled_diagnostics_do_not_copy_unsampled_k_heads():
    override=afm.AnimaAFMAttentionOverride(afm.AFMConfig(mode='observe',spectral_diag='sampled',diagnostic_max_heads=1))
    args=inputs(heads=8)
    real=torch.Tensor.index_select
    copied=[]
    def record(tensor,dim,index):
        if tensor is args[1] and dim==0:
            copied.append(tensor.shape[1])
        return real(tensor,dim,index)
    with patch.object(torch.Tensor,'index_select',record):
        override(original,*args,**options())
    assert not copied

@pytest.mark.parametrize('bad', ['bad', [[1,2,3]], [{}], [[0,4]]])
def test_malformed_candidate_collection_is_fail_closed(bad):
    info=afm.infer_spatial_shape(16,afm.AFMConfig(),{'afm_spatial_shape':[4,4],'afm_spatial_shape_candidates':bad})
    assert info.shape is None


def test_explicit_grid_cannot_override_authoritative_runtime_geometry():
    config=afm.AFMConfig(spatial_shape_mode='explicit_latent',latent_height=48,latent_width=48)
    info=afm.infer_spatial_shape(2304,config,{'afm_spatial_shape':[36,64]})
    assert info.shape is None
    assert info.reason=='spatial_shape_ambiguous'


def test_legacy_square_assumption_cannot_override_runtime_rectangle():
    info=afm.infer_spatial_shape(2304,afm.AFMConfig(spatial_shape_mode='square_only'),{'afm_spatial_shape':[36,64]})
    assert info.shape is None
    assert info.reason=='spatial_shape_ambiguous'


def test_managed_revisited_step_is_finalized_only_at_outer_completion():
    override=afm.AnimaAFMAttentionOverride(afm.AFMConfig(debug_level='summary',debug_format='jsonl'))
    override.managed_lifecycle=True
    records=[]
    with patch.object(override,'_emit_jsonl',side_effect=records.append):
        for sigma in (1.,.5,1.):
            kwargs=options();kwargs['transformer_options']['sigmas']=torch.tensor([sigma])
            override.begin_model_evaluation()
            override(original,*inputs(),**kwargs)
        assert not [r for r in records if r['record_type']=='step_final_summary']
        override.finalize()
    final=[r for r in records if r['record_type']=='step_final_summary']
    assert len(final)==2
    assert override.stats.finalized_steps=={0,1}


def test_wrapper_proves_equal_length_cross_attention_without_leaking_marker():
    override=afm.AnimaAFMAttentionOverride(afm.AFMConfig())
    class Cross(torch.nn.Module):
        def forward(self,q,k,v,heads,**kwargs):
            return override(original,q,k,v,heads,**kwargs)
    wrapper=afm.AnimaAFMBlockMetadataWrapper(Cross(),0,1,'blocks.0.cross_attn')
    q,k,v=(torch.randn(2,4,16,4) for _ in range(3))
    kwargs=options();before=set(kwargs['transformer_options'])
    wrapper(q,k,v,4,**kwargs)
    assert override.stats.edited_calls==1
    assert set(kwargs['transformer_options'])==before


def test_plain_metadata_does_not_prove_equal_length_is_cross_attention():
    override=afm.AnimaAFMAttentionOverride(afm.AFMConfig())
    kwargs=options();kwargs['transformer_options'].update(block_id='blocks.0.cross_attn',attention_kind='cross',_anima_afm_cross_attention=True)
    q,k,v=(torch.randn(2,4,16,4) for _ in range(3))
    override(original,q,k,v,4,**kwargs)
    assert override.stats.edited_calls==0
    assert override.stats.fallback_reasons['not_cross_attention']==1


def test_manual_attention_keeps_fp32_contract_inside_cpu_autocast():
    args=inputs()
    expected=afm.AnimaAFMAttentionOverride(afm.AFMConfig())(original,*args,**options())
    with torch.autocast('cpu',dtype=torch.bfloat16):
        actual=afm.AnimaAFMAttentionOverride(afm.AFMConfig())(original,*args,**options())
    assert actual.dtype==expected.dtype==torch.float32
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)


def test_observe_original_preserves_callers_autocast_context():
    observed=[]
    def backend(*args,**kwargs):
        observed.append(torch.is_autocast_enabled('cpu'));return original(*args,**kwargs)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        afm.AnimaAFMAttentionOverride(afm.AFMConfig(mode='observe'))(backend,*inputs(),**options())
    assert observed==[True]
