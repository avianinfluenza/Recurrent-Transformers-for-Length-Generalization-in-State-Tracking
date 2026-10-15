"""Independent controls used to resolve numerical failures in the trained probe."""

import torch

from scripts.analysis.analyze_causal_order_sensitivity import generate_pairs
from scripts.analysis.audit_causal_order_results import swap_transport
from src.data.collate import collate_fn
from src.original.model import FanRecurrentTransformer, OriginalModelConfig


def test_swap_hidden_transport_removes_the_second_iteration_slot_difference():
    torch.manual_seed(19)
    torch.set_num_threads(2)
    model=FanRecurrentTransformer(OriginalModelConfig(
        architecture="fan-recurrent",position_encoding="none",fan_input_format="atomic",
        d_model=32,n_heads=4,d_ff=64,num_layers=1,num_loops=10,
    )).double().eval()
    pairs=generate_pairs([2,4,19],5,31)[0]
    batch=collate_fn([p[s] for p in pairs for s in ('a','b')],input_format='atomic')
    before={k:v.clone() for k,v in model.state_dict().items()}
    results=swap_transport(model,batch,pairs)
    assert max(r['original_k2_relative_l2'] for r in results)>1e-4
    assert all(r['hidden_passed'] and r['normalized_passed'] and r['logits_passed'] for r in results)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
