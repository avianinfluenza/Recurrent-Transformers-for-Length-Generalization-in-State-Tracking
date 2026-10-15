#!/usr/bin/env python3
"""Audit float32 invariance failures in float64; summarize onset and SWAP mediation.

Run after analyze_causal_order_sensitivity.py, including when that command exits
nonzero for a tolerance violation. The original checks and tolerance are kept.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts.analysis.analyze_causal_order_sensitivity import (
    bidirectional_attention, distances, extract, gather_slots, invariance_checks,
    invariant, load_checkpoint, sample_metrics, sha256, write_csv, write_json,
)
from src.data.collate import collate_fn


@torch.inference_mode()
def swap_transport(model, batch:dict, pairs:list[dict]) -> list[dict]:
    """Copy A's K=1 SWAP states to their occurrence-matched B positions.

    Retain original B INIT and SLOT states and all original embeddings. This
    keeps the same multiset of K=1 SWAP states in the two K=2 SLOT contexts.
    """
    _,embedding,positions=model.prepare(batch["input_ids"],batch["attn_mask"])
    _,states=model.forward_all_loops(batch["input_ids"],batch["attn_mask"],batch["slot_pos"],
                                    num_loops=2,return_hidden_states=True)
    transported=states[0].clone()
    for i,pair in enumerate(pairs):
        transported[2*i+1,pair["b_positions_for_a"]]=states[0][2*i,5:5+pair["n_swaps"]]
    patched=model.recurrent_step(embedding,transported,batch["attn_mask"],positions)
    original_slots=gather_slots(states[1],batch["slot_pos"])
    patched_slots=gather_slots(patched,batch["slot_pos"])
    normalized=gather_slots(model.final_norm(patched)*batch["attn_mask"][...,None],batch["slot_pos"])
    patched_logits=model._classify(patched,batch["attn_mask"],batch["slot_pos"])
    original=distances(original_slots[0::2],original_slots[1::2])
    after=distances(patched_slots[0::2],patched_slots[1::2])
    passed=invariant(patched_slots[0::2],patched_slots[1::2],1e-10,1e-10)
    normalized_ok=invariant(normalized[0::2],normalized[1::2],1e-10,1e-10)
    logits_ok=invariant(patched_logits[0::2],patched_logits[1::2],1e-10,1e-10)
    return [dict(pair_id=p['pair_id'],n_swaps=p['n_swaps'],
                 original_k2_relative_l2=original['relative_l2'][i].mean().item(),
                 transported_k2_relative_l2=after['relative_l2'][i].mean().item(),
                 transported_k2_max_relative_l2=after['relative_l2'][i].max().item(),
                 hidden_passed=bool(passed[i].all()),normalized_passed=bool(normalized_ok[i].all()),
                 logits_passed=bool(logits_ok[i].all())) for i,p in enumerate(pairs)]


def summarize_onset(output:Path,manifest:dict) -> tuple[list[dict],dict]:
    groups=defaultdict(list)
    with (output/'per_sample_per_k.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['condition']=='causal':
                groups[(int(row['seed']),row['pair_id'])].append(row)
    onsets=[]
    for (seed,pair_id),rows in sorted(groups.items()):
        rows.sort(key=lambda r:int(r['k']))
        first_raw=None
        first_robust=None
        first_prediction=None
        first_both_correct=None
        for row in rows:
            l2=np.array(json.loads(row['hidden_l2_by_slot']))
            rel=np.array(json.loads(row['hidden_relative_l2_by_slot']))
            scale=np.divide(l2,rel,out=np.zeros_like(l2),where=rel!=0)
            if first_raw is None and (l2>manifest['atol']+manifest['rtol']*scale).any():
                first_raw=int(row['k'])
            # A second, explicitly declared descriptive threshold, not a p-value.
            if first_robust is None and (rel>1e-4).any():
                first_robust=int(row['k'])
            if first_prediction is None and row['prediction_a']!=row['prediction_b']:
                first_prediction=int(row['k'])
            if first_both_correct is None and int(row['both_exact']):
                first_both_correct=int(row['k'])
        onsets.append(dict(seed=seed,pair_id=pair_id,n_swaps=int(rows[0]['n_swaps']),
                           first_raw_difference_k=first_raw,first_raw_relative_above_1e4_k=first_robust,
                           first_prediction_difference_k=first_prediction,first_both_exact_k=first_both_correct))
    write_csv(output/'order_sensitivity_onset.csv',onsets)
    summary=dict(
        raw_difference_rule=manifest['tolerance'],
        robust_relative_rule='Any SLOT relative L2 > 1e-4 (descriptive, not a significance test)',
        raw_onset_histogram=dict(Counter(str(r['first_raw_difference_k']) for r in onsets)),
        robust_onset_histogram=dict(Counter(str(r['first_raw_relative_above_1e4_k']) for r in onsets)),
        prediction_onset_histogram=dict(Counter(str(r['first_prediction_difference_k']) for r in onsets)),
        total_seed_pairs=len(onsets),
    )
    write_json(output/'order_sensitivity_onset_summary.json',summary)
    return onsets,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'results/analysis/causal_order_sensitivity')
    parser.add_argument('--batch-pairs',type=int,default=25)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    output=args.results.resolve()
    manifest=json.loads((output/'execution_config.json').read_text())
    if (output/'postprocessing_config.json').exists():
        raise SystemExit('Audit outputs already exist; choose a fresh result directory.')
    started=time.perf_counter()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(manifest['eval_seed'])
    pairs=[json.loads(line) for line in (output/'paired_inputs.jsonl').read_text().splitlines()]
    checks=json.loads((output/'sanity_checks.json').read_text())
    bad=[r for r in checks if not r['passed']]
    if any(r['condition']=='restoration' for r in bad):
        raise SystemExit('Restoration failed: this is not a floating-point permutation discrepancy.')
    failing_groups={(r['seed'],r['n_swaps']) for r in bad}
    audit_checks,audit_rows,transport_rows=[],[],[]
    for meta in manifest['checkpoints']:
        model,_=load_checkpoint(Path(meta['path']),'cpu',torch.float64)
        seed=meta['seed']
        for n in manifest['lengths']:
            length_pairs=[p for p in pairs if p['n_swaps']==n]
            for start in range(0,len(length_pairs),args.batch_pairs):
                pp=length_pairs[start:start+args.batch_pairs]
                batch=collate_fn([p[s] for p in pp for s in ('a','b')],input_format='atomic')
                if (seed,n) in failing_groups:
                    original=extract(model,batch,manifest['max_loops'])
                    audit_checks.extend(invariance_checks(original,'causal',pp,seed,1e-10,1e-10))
                    with bidirectional_attention(model):
                        control=extract(model,batch,manifest['max_loops'])
                    audit_checks.extend(invariance_checks(control,'noncausal',pp,seed,1e-10,1e-10))
                    for condition,x in [('causal',original),('noncausal',control)]:
                        audit_rows.extend(sample_metrics(x,pp,seed,condition))
                    restored=extract(model,batch,manifest['max_loops'])
                    audit_checks.append(dict(seed=seed,condition='restoration',n_swaps=n,batch_start=start,
                                             passed=all(torch.equal(original[k],restored[k]) for k in original)))
                for row in swap_transport(model,batch,pp):
                    transport_rows.append(dict(seed=seed,**row))
            print(f'float64 audit/transport seed={seed} N={n} complete',flush=True)
    write_json(output/'float64_audit_checks.json',audit_checks)
    write_csv(output/'float64_audit_per_sample_per_k.csv',audit_rows)
    write_csv(output/'swap_transport_k2_float64.csv',transport_rows)
    _,onset_summary=summarize_onset(output,manifest)
    failed_double=[r for r in audit_checks if not r['passed']]
    transport_failures=[r for r in transport_rows if not all(r[k] for k in ('hidden_passed','normalized_passed','logits_passed'))]
    audit_summary=dict(
        original_dtype=manifest['dtype'],original_atol=manifest['atol'],original_rtol=manifest['rtol'],
        original_failed_checks=len(bad),original_failures_by_tensor=dict(Counter(r.get('tensor','restoration') for r in bad)),
        rechecked_seed_length_groups=sorted([list(g) for g in failing_groups]),
        scope='All 200 pairs (or configured count) and all K in every affected seed/length, both masks',
        audit_dtype='float64',audit_device='cpu',audit_atol=1e-10,audit_rtol=1e-10,
        audit_checks=len(audit_checks),audit_failed_checks=len(failed_double),
        audit_max_noncausal_hidden_relative_l2=max((r['max_relative_l2'] for r in audit_checks if r.get('tensor')=='hidden' and r['condition']=='noncausal'),default=None),
        audit_max_noncausal_probability_relative_l2=max((r['max_relative_l2'] for r in audit_checks if r.get('tensor')=='probabilities' and r['condition']=='noncausal'),default=None),
        transport_seed_pairs=len(transport_rows),transport_failed_pairs=len(transport_failures),
        transport_original_k2_mean_relative_l2=float(np.mean([r['original_k2_relative_l2'] for r in transport_rows])),
        transport_after_k2_mean_relative_l2=float(np.mean([r['transported_k2_relative_l2'] for r in transport_rows])),
        transport_after_max_relative_l2=max(r['transported_k2_max_relative_l2'] for r in transport_rows),
        interpretation=('Float32 tolerance violations resolve at much tighter float64 tolerance; consistent with reduction rounding amplified by recurrence/softmax. Not evidence against algebraic permutation invariance.'
                        if bad and not failed_double else 'See detailed checks.'),
    )
    write_json(output/'float64_audit_summary.json',audit_summary)
    # Append a clear audit to the original report. Preserve all original failures.
    with (output/'aggregate.csv').open() as stream:
        aggregate=list(csv.DictReader(stream))
    with (output/'aggregate_by_seed.csv').open() as stream:
        by_seed=list(csv.DictReader(stream))
    report=output/'report_ko.md'
    section=["", "## 추가 검증: 수치 오차, 최초 차이, SWAP 전달 경로", "",
             f"사전에 정한 float32 허용 오차를 넘은 검사: **{len(bad)}개**. 이를 통과로 바꾸거나 허용 오차를 넓히지 않았다. tensor별 실패 건수: `{audit_summary['original_failures_by_tensor']}`. 원본 `sanity_checks.json`에 그대로 남겼다.",
             f"영향받은 (seed,N)={audit_summary['rechecked_seed_length_groups']}의 **모든 pair·모든 K·두 attention 조건**을 CPU float64로 재평가했다. 훨씬 엄격한 atol=rtol=1e-10에서 {len(audit_checks)}개 검사 중 실패 **{len(failed_double)}개**였다."]
    if audit_summary['audit_max_noncausal_hidden_relative_l2'] is not None:
        section += [f"float64 noncausal의 최대 SLOT hidden relative L2={audit_summary['audit_max_noncausal_hidden_relative_l2']:.3g}, probability relative L2={audit_summary['audit_max_noncausal_probability_relative_l2']:.3g}. float32의 차이가 float64에서 줄어드는 결과는 순열에 따른 합산 순서 오차가 recurrence와 softmax를 거치며 증폭되는 설명과 일치한다. 구조적 불변성 실패라는 근거는 관측하지 못했다."]
    section += ["", f"**최초 차이:** {onset_summary['total_seed_pairs']}개 seed×pair에서 raw hidden이 원래 수치 오차 기준을 처음 넘은 K의 분포는 `{onset_summary['raw_onset_histogram']}`이다. 보조 기준(relative L2>1e-4)의 분포는 `{onset_summary['robust_onset_histogram']}`이다. `order_sensitivity_onset.csv`에 각 pair의 최초 hidden 차이·최초 예측 차이·최초 양쪽 exact 정답 K를 저장했다.",
                "", "**SWAP hidden 전달 통제:** 동일 causal 모델에서 B의 K=1 SWAP hidden만 A의 identity/occurrence 대응 hidden으로 교체하고, 원래 B embedding·INIT/SLOT hidden을 유지한 채 K=2를 계산했다. 이는 학습/정확도 비교가 아니라 정보 전달 경로의 진단이다.",
                f"전체 {len(transport_rows)}개 seed×pair에서 원래 K=2 평균 raw relative L2={audit_summary['transport_original_k2_mean_relative_l2']:.4g}, SWAP hidden 교체 후={audit_summary['transport_after_k2_mean_relative_l2']:.3g}, 최대 SLOT relative L2={audit_summary['transport_after_max_relative_l2']:.3g}. raw/normalized/logits가 float64 허용 오차를 넘은 pair는 {len(transport_failures)}개였다. K=1 SWAP hidden의 context 차이를 제거하면 K=2 SLOT 차이가 사라지는지 직접 검사한 결과이다.",
                "", "**구조적 설명:** noncausal NoPE의 attention·token별 LayerNorm/FFN·embedding reinjection은 토큰 순열에 대해 equivariant이다. 이를 recurrence마다 귀납하면 모든 K에서 equivariance가 유지된다. SWAP만 움직이는 순열은 SLOT 위치를 고정하므로 SLOT 출력은 invariant이다. causal L1의 K=1 SLOT은 동일 query와 동일한 prefix token multiset을 보므로 같은 결론이 성립하지만, SWAP 위치별 prefix가 달라져 K=1 SWAP hidden의 multiset은 이후 달라질 수 있다. 다음 attention에서 SLOT이 이 hidden들을 읽을 수 있다.",
                "", "### K=N에서의 정답 추적 (seed 평균 ± seed SD)", "",
                "|N|SLOT raw rel L2|SLOT disagreement|exact-match|slot accuracy|양쪽 모두 exact|", "|---:|---:|---:|---:|---:|---:|"]
    for r in aggregate:
        if r['condition']=='causal' and r['n_swaps']==r['k']:
            section.append(f"|{r['n_swaps']}|{float(r['hidden_relative_l2']):.4g}|{float(r['prediction_disagreement']):.2%}|{float(r['exact_match']):.2%} ± {float(r['exact_match_seed_sd']):.2%}|{float(r['slot_accuracy']):.2%}|{float(r['both_exact']):.2%}|")
    section += ["", "seed별 N=19 결과:", "", "|seed|K|raw rel L2|disagreement|exact-match|양쪽 모두 exact|", "|---:|---:|---:|---:|---:|---:|"]
    for r in by_seed:
        if r['condition']=='causal' and int(r['n_swaps'])==19 and int(r['k']) in (2,19,24):
            section.append(f"|{r['seed']}|{r['k']}|{float(r['hidden_relative_l2']):.4g}|{float(r['prediction_disagreement']):.2%}|{float(r['exact_match']):.2%}|{float(r['both_exact']):.2%}|")
    section += ["", "추가 검증 재실행 명령(원본 분석 이후):", "", "```sh",
                f".venv/bin/python scripts/analysis/audit_causal_order_results.py --results {output}", "```", ""]
    report.write_text(report.read_text()+"\n".join(section))
    import shlex
    write_json(output/'postprocessing_config.json',dict(
        created_utc=datetime.now(timezone.utc).isoformat(),command=shlex.join([sys.executable,*sys.argv]),
        source_sha256=sha256(Path(__file__)),original_execution_config_sha256=sha256(output/'execution_config.json'),
        threads=args.threads,batch_pairs=args.batch_pairs,elapsed_seconds=time.perf_counter()-started,
        audit_summary=audit_summary,onset_summary=onset_summary,
        all_work_complete=not failed_double and not transport_failures,
    ))
    print(json.dumps(audit_summary,indent=2),flush=True)
    if failed_double or transport_failures:
        raise SystemExit('Float64 invariance/mediation still fails. Inspect results before interpretation.')


if __name__=='__main__':
    main()
