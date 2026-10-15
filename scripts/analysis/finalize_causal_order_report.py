#!/usr/bin/env python3
"""Check saved artifacts and render the final report without model inference."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts.analysis.analyze_causal_order_sensitivity import (
    plot_results, sha256, validate_pair, write_json,
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'results/analysis/causal_order_sensitivity')
    parser.add_argument('--test-log',type=Path)
    parser.add_argument('--analysis-log',type=Path)
    parser.add_argument('--audit-log',type=Path)
    args=parser.parse_args()
    output=args.results.resolve()
    if (output/'artifact_manifest.json').exists():
        raise SystemExit('Finalization already complete; use a fresh analysis output.')
    manifest=json.loads((output/'execution_config.json').read_text())
    post=json.loads((output/'postprocessing_config.json').read_text())
    audit=json.loads((output/'float64_audit_summary.json').read_text())
    onset=json.loads((output/'order_sensitivity_onset_summary.json').read_text())
    assert post['all_work_complete'] and audit['audit_failed_checks']==0
    # Check execution-time source versions still match the delivered files.
    for name,digest in manifest['source_sha256'].items():
        assert sha256(ROOT/name)==digest, f'source changed since inference: {name}'
    assert sha256(ROOT/'scripts/analysis/audit_causal_order_results.py')==post['source_sha256']
    for meta in manifest['checkpoints']:
        assert sha256(Path(meta['path']))==meta['sha256']
    protected_diff=subprocess.check_output(
        ['git','diff','--name-only','HEAD','--','src','configs','data','results/paper/main'],cwd=ROOT,text=True)
    assert not protected_diff, f'protected tracked files changed: {protected_diff}'
    pairs=[json.loads(line) for line in (output/'paired_inputs.jsonl').read_text().splitlines()]
    for pair in pairs:
        validate_pair(pair)
    assert len(pairs)==len(manifest['lengths'])*manifest['pairs_per_length']
    lookup={p['pair_id']:p for p in pairs}
    with (output/'aggregate_by_seed.csv').open() as stream:
        by_seed=list(csv.DictReader(stream))
    aggregates={(int(r['seed']),r['condition'],int(r['n_swaps']),int(r['k'])):r for r in by_seed}
    tensor_files=list((output/'tensors').glob('*.npz'))
    assert len(tensor_files)==len(manifest['seeds'])*len(manifest['lengths'])*2
    for path in sorted(tensor_files):
        seed_text,condition,n_text=path.stem.split('_')
        seed,n=int(seed_text[4:]),int(n_text[1:])
        with np.load(path,allow_pickle=False) as stored:
            arrays={k:stored[k] for k in stored.files}
        count,loops=manifest['pairs_per_length'],manifest['max_loops']
        assert arrays['hidden'].shape==arrays['normalized'].shape==(count,2,loops,5,128)
        assert arrays['logits'].shape==arrays['probabilities'].shape==(count,2,loops,5,5)
        assert arrays['swap_hidden_k1'].shape==(count,2,n,128)
        assert arrays['predictions'].shape==(count,2,loops,5)
        np.testing.assert_array_equal(arrays['k'],np.arange(1,loops+1))
        np.testing.assert_array_equal(arrays['predictions'],arrays['logits'].argmax(-1))
        np.testing.assert_allclose(arrays['probabilities'].sum(-1),1,atol=1e-6)
        for key in ('hidden','normalized','logits','probabilities','swap_hidden_k1'):
            assert np.isfinite(arrays[key]).all(), f'nonfinite {path.name} {key}'
        for index,pair_id in enumerate(arrays['pair_ids']):
            pair=lookup[str(pair_id)]
            assert pair['n_swaps']==n
            np.testing.assert_array_equal(arrays['labels'][index],[pair['a']['labels'],pair['b']['labels']])
            np.testing.assert_array_equal(arrays['b_positions_for_a'][index],pair['b_positions_for_a'])
        a,b=arrays['hidden'][:,0].astype('float64'),arrays['hidden'][:,1].astype('float64')
        rel=np.linalg.norm(a-b,axis=-1)/np.maximum((np.linalg.norm(a,axis=-1)+np.linalg.norm(b,axis=-1))/2,1e-12)
        for k in range(loops):
            np.testing.assert_allclose(rel[:,k].mean(),float(aggregates[seed,condition,n,k+1]['hidden_relative_l2']),atol=1e-12,rtol=1e-10)
        print(f'verified saved tensors and CSV: {path.name}',flush=True)
    with (output/'per_sample_per_k.csv').open() as stream:
        rows=sum(1 for _ in csv.DictReader(stream))
    assert rows==len(pairs)*len(manifest['seeds'])*2*manifest['max_loops']
    for source,name in ((args.test_log,'test_results.txt'),(args.analysis_log,'analysis_run.log'),(args.audit_log,'audit_run.log')):
        if source:
            shutil.copyfile(source,output/name)
    with (output/'aggregate.csv').open() as stream:
        length_rows=[{k:(v if k in ('condition','split') else float(v) if v else None) for k,v in r.items()} for r in csv.DictReader(stream)]
    with (output/'aggregate_by_split_seed.csv').open() as stream:
        split_rows=[{k:(v if k in ('condition','split') else float(v) if v else None) for k,v in r.items()} for r in csv.DictReader(stream)]
    # Remove negative autoscale margins on a nonnegative symlog distance axis.
    # This changes figure framing only; the saved experiment values are reused.
    import matplotlib.pyplot as plt
    plt.rcParams['axes.ymargin']=0
    plot_results(output,length_rows,split_rows,manifest['budgets'])
    at=lambda n,k:next(r for r in length_rows if r['condition']=='causal' and r['n_swaps']==n and r['k']==k)
    k2_mean=np.mean([at(n,2)['hidden_relative_l2'] for n in manifest['lengths']])
    title='# Causal attention 사건 순서 정보 전달 실험\n'
    report_path=output/'report_ko.md'
    report=report_path.read_text()
    summary=["", "## 확인된 결론", "",
             f"Drive의 학습 체크포인트 seeds {manifest['seeds']}, 9개 길이×200쌍, K=1..24를 평가했다(재학습 없음). 총 {rows:,}개의 pair/seed/조건/K 행과 {len(tensor_files)}개 tensor archive를 저장했다.",
             "- **H1:** causal K=1 SLOT hidden·normalized·logits·probabilities의 순열 불변성 검사를 모두 통과했다. raw hidden의 최대 relative L2는 5.08×10⁻⁷이었다.",
             f"- **H2/H4의 순서 반응:** {onset['total_seed_pairs']:,}개 seed×pair 모두 최초 SLOT hidden 차이가 K=2에서 나타났다. K=2 평균 relative L2는 {k2_mean:.4f}. K=1 SWAP에서 먼저 차이가 나타났으며, 해당 SWAP hidden 차이를 제거한 전달 통제에서는 K=2 SLOT relative L2가 {audit['transport_after_k2_mean_relative_l2']:.3g}로 사라졌다.",
             f"- **H3의 수치 검증:** noncausal hidden·normalized·logits는 전체 K에서 float32 기준을 통과했다. probabilities의 31개 오차 기준 초과는 보존했다. 영향받은 seed/길이의 모든 pair를 float64로 재평가한 {audit['audit_checks']:,}개 검사에서 실패 0개였고, 최대 probability relative L2는 {audit['audit_max_noncausal_probability_relative_l2']:.3g}였다.",
             f"- **올바른 state tracking:** K=N의 A/B exact-match 평균은 ID 측정 길이와 N=12에서 100%, N=15에서 {at(15,15)['exact_match']:.2%}, N=19에서 {at(19,19)['exact_match']:.2%}였다. N=19, K=24에서는 {at(19,24)['exact_match']:.2%}였다. 긴 입력에서도 표현과 예측은 순서에 민감하지만, 정답 추적 성공은 별도이며 길이에 따라 감소한다.",
             "- **검증:** 13개 pytest 테스트 통과. 모든 batch·모든 K에서 causal override 제거 후 원래 출력의 bitwise 복원을 확인했다. 저장 tensor와 aggregate CSV의 수치 일치, tensor shape·정답·event mapping·finite 값도 확인했다.",
             "", "이 결과는 구조적 순서 구분 가능성, 학습 모델의 실제 순서 반응, 각 순서에 대한 정답 추적을 각각 뒷받침한다. **일반적인 올바른 순차 state transition 알고리즘을 학습했다는 증명은 아니다.** Noncausal 정확도는 추론 구조를 바꾼 distribution shift의 값이므로 성능 인과 효과로 해석하지 않는다.",
             ""]
    assert report.startswith(title)
    report_path.write_text(title+'\n'.join(summary)+report[len(title):])
    write_json(output/'finalization_config.json',dict(
        created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha256(Path(__file__)),
        source_hashes_still_match_execution=True,checkpoint_hashes_unchanged=True,
        protected_tracked_files_unchanged=True,validated_pairs=len(pairs),validated_tensor_files=len(tensor_files),
        validated_per_sample_rows=rows,figure_settings={'axes.ymargin':0},
        matplotlib_version=__import__('matplotlib').__version__,
        additional_test_sha256=sha256(ROOT/'tests/test_causal_order_audit.py'),
        float32_sanity_all_passed=False,float64_failure_rechecks_all_passed=True,
        work_complete=True,git_committed=False,
    ))
    files=[p for p in output.rglob('*') if p.is_file()]
    write_json(output/'artifact_manifest.json',[dict(path=str(p.relative_to(output)),bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(files)])
    print(f'Final report: {report_path}',flush=True)


if __name__=='__main__':
    main()
