# 결과 안내

한국어 전체 보고서는 [report_ko.md](report_ko.md)를 참조하세요.

- 학습 checkpoint seeds 0/1/2, 9개 길이, 길이당 200쌍, K=1..24.
- 원본 representation·normalized representation·logits·probabilities·예측은 `tensors/*.npz`에 보존했습니다. 축은 `[pair, A/B, K, SLOT, feature]`입니다.
- `per_sample_per_k.csv`: 259,200행. `aggregate_by_seed.csv`, `aggregate.csv`: seed별 및 seed 평균 집계.
- `paired_inputs.jsonl`: 입력·정답·사건 occurrence ID·대응 위치.
- `figure1/2/3.png` 및 `.pdf`: 요청한 세 그림.
- `representative_samples.csv`: 각 길이의 첫 생성 pair에 대한 전체 K 경로.
- `order_sensitivity_onset.csv`: pair별 최초 hidden/예측 차이 K.
- `swap_hidden_k1.csv`: identity/occurrence 대응 첫 iteration SWAP 차이.
- `swap_transport_k2_float64.csv`: K=1 SWAP hidden 차이를 제거한 추가 전달 통제.
- `test_results.txt`: 13개 pytest 테스트 통과 기록.
- `execution_config.json`: 최초 float32 실험 설정과 checkpoint/source SHA256.
- `postprocessing_config.json`, `float64_audit_*.json/csv`: 수치 오차 재검증.
- `finalization_config.json`: 최종 완료 상태와 저장 tensor/CSV 검증 결과.
- `artifact_manifest.json`: 결과 파일의 SHA256 및 크기.
- `environment_freeze.txt`: 실행 환경의 Python package versions.

## 완료 상태 읽는 법

원본 `execution_config.json`의 `complete=false`는 엄격한 최초 float32 불변성
검사에서 noncausal probability 31개가 허용 오차를 넘었다는 기록입니다.
이 실패와 원래 tolerance를 그대로 보존했습니다. hidden·normalized·logits와
causal K=1 검사는 모두 통과했습니다.

영향받은 seed/길이의 모든 pair를 CPU float64로 재평가한 6,464개 검사에서는
더 엄격한 tolerance(atol=rtol=1e-10)로 모두 통과했습니다. 추가 전달 통제와
artifact 검증도 끝났으며, 최종 완료 상태는 `finalization_config.json`의
`work_complete=true`입니다. 구조적 실패와 float32 수치 오차를 구분한 결과입니다.

## 재실행

저장소 루트에서 실행하세요. 출력 경로는 새 디렉터리를 지정해야 합니다.
체크포인트는 `checkpoints/causal_order_sensitivity/seed{0,1,2}/checkpoint.pt`,
학습 인자는 각 sibling `args.json`에 있습니다. Drive 출처는
`checkpoint_sources.json`에 기록했습니다. 재학습은 수행하지 않습니다.

```sh
.venv/bin/python scripts/analysis/analyze_causal_order_sensitivity.py \
  --checkpoints checkpoints/causal_order_sensitivity/seed{0,1,2}/checkpoint.pt \
  --output /tmp/causal_order_reproduction
```

위 명령은 모든 결과를 저장한 후 엄격한 float32 검사 실패가 있으면 nonzero로
종료합니다. 그 경우 아래 재검증 단계로 원인을 확인하세요.

```sh
.venv/bin/python scripts/analysis/audit_causal_order_results.py \
  --results /tmp/causal_order_reproduction
.venv/bin/python -m pytest \
  tests/test_causal_order_sensitivity.py tests/test_causal_order_audit.py -rA \
  > /tmp/causal_order_reproduction_tests.txt
.venv/bin/python scripts/analysis/finalize_causal_order_report.py \
  --results /tmp/causal_order_reproduction \
  --test-log /tmp/causal_order_reproduction_tests.txt
```

기존 모델·학습 코드·dataset·`results/paper/main/`과 원본 체크포인트는
수정하지 않았습니다. Git commit은 생성하지 않았습니다.
