"""Structural, data-integrity, and extraction tests for the independent probe."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from scripts.analysis.analyze_causal_order_sensitivity import (
    bidirectional_attention, distances, event_ids, extract, generate_pairs,
    invariant, load_checkpoint, simulate, validate_pair,
)
from src.data.collate import collate_fn
from src.original.model import FanRecurrentTransformer, OriginalModelConfig


@pytest.fixture
def model():
    torch.manual_seed(17)
    torch.set_num_threads(2)
    return FanRecurrentTransformer(OriginalModelConfig(
        architecture="fan-recurrent", position_encoding="none", fan_input_format="atomic",
        d_model=32, n_heads=4, d_ff=64, num_layers=1, num_loops=10, dropout=0.0,
    )).eval()


def batch_for(pairs):
    return collate_fn([p[s] for p in pairs for s in ("a", "b")], input_format="atomic")


def test_pairs_are_deterministic_unique_and_have_different_ground_truth():
    lengths=[2,3,4,6,8,10,12,15,19]
    pairs,stats=generate_pairs(lengths,20,20261015)
    assert (pairs,stats)==generate_pairs(lengths,20,20261015)
    assert all(sum(p["n_swaps"]==n for p in pairs)==20 for n in lengths)
    for pair in pairs:
        validate_pair(pair)
    assert simulate([0,1,2,3,4],[[0,1],[1,2]]) != simulate([0,1,2,3,4],[[1,2],[0,1]])
    assert simulate([0,1,2,3,4],[[0,1],[2,3]]) == simulate([0,1,2,3,4],[[2,3],[0,1]])


def test_validator_rejects_wrong_labels_and_multiset():
    pair=generate_pairs([4],1,9)[0][0]
    wrong=copy.deepcopy(pair)
    wrong["b"]["labels"]=wrong["a"]["labels"]
    with pytest.raises(AssertionError):
        validate_pair(wrong)
    wrong=copy.deepcopy(pair)
    wrong["b"]["swaps"][0]=[0,0]
    with pytest.raises(AssertionError):
        validate_pair(wrong)


def test_repeated_events_are_matched_by_identity_and_occurrence():
    swaps=[[0,1],[1,2],[0,1],[1,2]]
    ids=event_ids(swaps)
    assert ids==["SWAP_0_1#0","SWAP_1_2#0","SWAP_0_1#1","SWAP_1_2#1"]
    permutation=[2,0,3,1]
    identities_b=[ids[i] for i in permutation]
    corresponding=[identities_b.index(identity) for identity in ids]
    assert corresponding==[1,3,0,2]
    # Duplicate events carry their original ID; they are not rematched by position.
    assert all(swaps[j]==[swaps[i] for i in permutation][corresponding[j]] for j in range(4))


def test_causal_k1_slot_invariance_and_swap_difference_precedes_slots(model):
    pairs=generate_pairs([2,4,19],5,13)[0]
    # Mixed lengths deliberately exercise PAD and slot_pos handling.
    batch=batch_for(pairs)
    x=extract(model,batch,24)
    for key in ("hidden","normalized","logits","probabilities"):
        assert invariant(x[key][0::2,0],x[key][1::2,0],1e-5,1e-5).all()
    assert (distances(x["hidden"][0::2,1],x["hidden"][1::2,1])["relative_l2"]>1e-4).any()
    pair=pairs[0]
    a=x["first_full_hidden"][0,5:7]
    b=x["first_full_hidden"][1,pair["b_positions_for_a"]]
    assert (distances(a,b)["relative_l2"]>1e-4).any()
    # No backward information path into the identical INIT prefix.
    assert torch.equal(x["first_full_hidden"][0::2,:5],x["first_full_hidden"][1::2,:5])


def test_noncausal_all_k_equivariance_including_occurrences_and_padding(model):
    pairs=generate_pairs([3,19],5,17)[0]
    batch=batch_for(pairs)
    with bidirectional_attention(model):
        x=extract(model,batch,24)
    for key in ("hidden","normalized","logits","probabilities"):
        assert invariant(x[key][0::2],x[key][1::2],1e-5,1e-5).all()
    for i,pair in enumerate(pairs):
        a=x["first_full_hidden"][2*i,5:5+pair["n_swaps"]]
        b=x["first_full_hidden"][2*i+1,pair["b_positions_for_a"]]
        assert invariant(a,b,1e-5,1e-5).all()


@pytest.mark.parametrize("raise_exception",[False,True])
def test_override_restores_identical_original_forward_even_on_exception(model,raise_exception):
    batch=batch_for(generate_pairs([3],3,1)[0])
    before=extract(model,batch,24)
    state_before={k:v.clone() for k,v in model.state_dict().items()}
    try:
        with bidirectional_attention(model):
            assert all(len(layer.attention._forward_pre_hooks)==1 for layer in model.shared_layers)
            _=extract(model,batch,24)
            if raise_exception:
                raise RuntimeError("intentional exception")
    except RuntimeError:
        assert raise_exception
    after=extract(model,batch,24)
    assert all(torch.equal(before[k],after[k]) for k in before)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state_before.items())
    assert all(not layer.attention._forward_pre_hooks for layer in model.shared_layers)


def test_normalized_representation_is_the_actual_classifier_input(model):
    batch=batch_for(generate_pairs([4],3,7)[0])
    actual=[]
    handle=model.classifier.register_forward_pre_hook(lambda module,args: actual.append(args[0].clone()))
    try:
        x=extract(model,batch,4)
    finally:
        handle.remove()
    assert torch.equal(torch.stack(actual,dim=1),x["normalized"])
    with torch.inference_mode():
        independently_classified=torch.stack([model.classifier(x["normalized"][:,k]) for k in range(4)],dim=1)
    assert torch.equal(independently_classified,x["logits"])
    assert torch.equal(x["predictions"],x["logits"].argmax(-1))
    torch.testing.assert_close(x["probabilities"].sum(-1),torch.ones_like(x["predictions"],dtype=torch.float))


def test_order_sensitivity_is_not_guaranteed_for_every_parameter_setting(model):
    # A valid parameter setting with no cross-token information transfer.
    with torch.no_grad():
        for layer in model.shared_layers:
            layer.attention.output.weight.zero_()
            layer.attention.output.bias.zero_()
    x=extract(model,batch_for(generate_pairs([2],3,2)[0]),24)
    assert torch.equal(x["hidden"][0::2],x["hidden"][1::2])


def test_distance_definition_is_symmetric_and_zero_safe():
    a=torch.tensor([[3.,4.],[0.,0.]])
    b=torch.tensor([[0.,0.],[0.,0.]])
    d=distances(a,b)
    torch.testing.assert_close(d["relative_l2"],torch.tensor([2.,0.],dtype=torch.float64))
    assert torch.equal(d["relative_l2"],distances(b,a)["relative_l2"])


def test_loader_refuses_missing_and_nonpaper_checkpoints(tmp_path:Path,model):
    with pytest.raises(FileNotFoundError,match="trained checkpoint required"):
        load_checkpoint(tmp_path/"missing.pt","cpu",torch.float32)
    path=tmp_path/"wrong.pt"
    torch.save(dict(config=model.config.to_dict(),model_state=model.state_dict()),path)
    with pytest.raises(ValueError,match="does not match paper"):
        load_checkpoint(path,"cpu",torch.float32)


def test_trained_checkpoints_match_paper_config_and_strict_load():
    root=Path(__file__).resolve().parents[1]
    paths=list((root/"checkpoints/causal_order_sensitivity").glob("seed*/checkpoint.pt"))
    if not paths:
        pytest.skip("local trained checkpoint not present; never substitute random weights")
    for path in sorted(paths):
        model,metadata=load_checkpoint(path,"cpu",torch.float32)
        assert metadata["config_matches_paper"] and metadata["strict_state_dict_loading"]
        assert not metadata["missing_training_metadata"]
        assert metadata["seed"] in (0,1,2)
        assert metadata["epoch"]==100000
        assert model.config.num_layers==1
