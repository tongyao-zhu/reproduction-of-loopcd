"""Logit oracle, cache isolation, native RNG, and native generation checks."""
import pytest
import torch
from test_huginn import model, FakeHuginn, readout
from loopcd_repro.huginn_logits import HuginnLogitsConfig as Config, HuginnLogitsGuidance as Guide


@pytest.mark.parametrize('config',[Config('baseline'),Config('fixed',omega=0),Config('adaptive',omega_cap=0)])
def test_disabled_exact_native(model,config):
    ids=torch.tensor([[3,4,5]])
    torch.manual_seed(31);expected=model(ids,num_steps=16).logits;state=torch.random.get_rng_state()
    torch.manual_seed(31)
    with Guide(model,config) as adapter:
        result=model(ids)
        assert adapter.last_observation['extra_coda_passes']==0
    assert torch.equal(expected,result.logits)
    assert torch.equal(state,torch.random.get_rng_state())


@pytest.mark.parametrize('mode',['fixed','adaptive'])
def test_complete_coda_oracle_and_rng(model,mode):
    ids=torch.tensor([[3,4,5]])
    states=[]
    hook=model.transformer.core_block[-1].register_forward_hook(lambda m,i,o:states.append(o.detach().clone()))
    torch.manual_seed(42);strong=model(ids,num_steps=16).logits;rng=torch.random.get_rng_state();hook.remove()
    early=readout(model,states[0])
    config=Config(mode)
    if mode=='fixed':strength=.2
    else:
        ps=strong.double().softmax(-1).topk(2,dim=-1).values
        strength=.25*(1-(ps[...,:1]-ps[...,1:2]))
    expected=strong.double()+strength*(strong.double()-early.double())
    torch.manual_seed(42)
    with Guide(model,config) as adapter:
        actual=model(ids)
        assert adapter.last_observation['lm_head_calls']==2
        assert adapter.last_observation['coda_layer_calls']==2*len(model.transformer.coda)
        assert adapter.last_observation['executed_physical_loops']==list(range(1,17))
    torch.testing.assert_close(actual.logits.double(),expected,atol=2e-6,rtol=2e-6)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert 'forward' not in model.__dict__


def cache_equal(left,right):
    assert left.get_seq_length()==right.get_seq_length()
    for name in ('key_cache','value_cache'):
        a,b=getattr(left,name),getattr(right,name)
        assert a.keys()==b.keys()
        for layer in a:
            assert a[layer].keys()==b[layer].keys()
            for pos in a[layer]:assert torch.equal(a[layer][pos],b[layer][pos])


def test_cached_reference_history_matches_full_and_preserves_strong(model):
    if isinstance(model,FakeHuginn):pytest.skip('Native cache value/state indexing tested on actual Raven class')
    ids=torch.tensor([[3,4,5,6]])
    initial=torch.randn((1,4,model.config.n_embd))
    with Guide(model,Config()) as adapter:
        full=model(ids,input_states=initial).logits
        main=adapter.new_cache();outputs=[]
        for start,end in ((0,2),(2,3),(3,4)):
            output=model(ids[:,start:end],input_states=initial[:,start:end],use_cache=True,past_key_values=main,
                         cache_position=torch.arange(start,end))
            outputs.append(output.logits)
            weak=main._loopcd_reference_coda_cache
            assert weak is not main and weak.get_seq_length()==end
            assert set(weak.key_cache)=={-1,-2}
            assert all(set(slot)==set(range(end)) for slot in weak.key_cache.values())
    torch.testing.assert_close(torch.cat(outputs,1),full,atol=2e-5,rtol=2e-5)
    # Strong KV is exactly the ordinary native fixed-depth KV, including coda.
    native=Guide(model,Config('baseline')).new_cache()
    for start,end in ((0,2),(2,3),(3,4)):
        model(ids[:,start:end],input_states=initial[:,start:end],num_steps=16,use_cache=True,past_key_values=native,
              cache_position=torch.arange(start,end))
    cache_equal(main,native)


def test_missing_weak_history_refused(model):
    if isinstance(model,FakeHuginn):pytest.skip('Native cache required')
    with Guide(model,Config()) as adapter:
        cache=adapter.new_cache()
        model(torch.tensor([[3,4]]),use_cache=True,past_key_values=cache)
        del cache._loopcd_reference_coda_cache
        with pytest.raises(ValueError,match='no reference-coda history'):
            model(torch.tensor([[5]]),use_cache=True,past_key_values=cache,cache_position=torch.tensor([2]))


def test_cap_change_cannot_reuse_cache(model):
    cache=Guide(model,Config()).new_cache()
    with Guide(model,Config(omega_cap=.5)):
        with pytest.raises(ValueError,match='different or unknown'):
            model(torch.tensor([[3,4]]),use_cache=True,past_key_values=cache)


def test_restoration_on_reference_failure(model):
    original=model.forward;heads=[]
    def fail_second(module,inputs):
        heads.append(1)
        if len(heads)==2:raise RuntimeError('reference failure')
    hook=model.lm_head.register_forward_pre_hook(fail_second)
    with pytest.raises(RuntimeError,match='reference failure'):
        with Guide(model,Config()):model(torch.tensor([[3,4]]))
    hook.remove()
    assert model.forward==original
    assert not model.transformer.core_block[-1]._forward_hooks
    assert not model.lm_head._forward_hooks
    assert all(not layer._forward_pre_hooks for layer in model.transformer.coda)


def test_native_generation_uses_separate_cache(model):
    if isinstance(model,FakeHuginn):pytest.skip('Native generation required')
    with Guide(model,Config()) as adapter:
        result=model.generate(torch.tensor([[3,4]]),max_new_tokens=2,min_new_tokens=2,do_sample=False,
                              use_cache=True,pad_token_id=30,eos_token_id=2)
        assert result.shape==(1,4)
        assert adapter.last_observation['reference_cache_separate']
        assert adapter.last_observation['reference_cache_length']==3
        with pytest.raises(ValueError):model.generate(torch.tensor([[3,4]]),continuous_compute=True)
