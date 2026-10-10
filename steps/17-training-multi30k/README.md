# Multi30k 독→영 번역 학습

- 원본: [시스템 학습](https://dev-jonghoonpark.github.io/annotated-transformer-modern/#training-the-system)

이제 진짜로 학습한다. 지금까지 만든 것이 모두 여기 모인다.

**`train_worker`**: GPU 하나에서 도는 학습 과정이다.
1. `make_model`로 6층 모델을 만들어 GPU에 올린다(분산 학습이면 `DDP`로 감싼다).
2. `LabelSmoothing(smoothing=0.1)`, `create_dataloaders`, Adam(β=(0.9, 0.98), ε=1e-9) + `rate` 스케줄을 준비한다.
3. epoch마다 `run_epoch`로 학습하고 체크포인트를 저장한 뒤 검증 손실을 잰다.

**`train_distributed_model`/`train_model`**: GPU가 여러 개면 `mp.spawn`으로 GPU마다 `train_worker`를 하나씩 띄운다.

**`load_trained_model`**: 학습 설정이 여기 있다. 배치 32, `accum_iter=10`(10배치마다 한 번 업데이트하므로 실질 배치 320), warmup 3000, 8 epoch. `multi30k_model_final.pt`가 이미 있으면 학습을 건너뛴다.

**`check_outputs`**: 검증 문장을 `greedy_decode`로 번역해서 원문·정답·모델 출력을 나란히 찍는다.

> 원본의 `load_trained_model`은 `vocab_src` 같은 노트북 전역 변수를 그대로 가져다 쓴다. 파일 하나로 import할 수 있도록 인자로 받게만 바꿨다. `train_distributed_model` 첫 줄의 `from the_annotated_transformer import train_worker`(노트북에서 `mp.spawn`에 함수를 넘기려고 넣은 줄)도 뺐다.

## 실행 결과에서 볼 것

- 손실이 7.6에서 시작해 epoch 7에 1점대로 내려간다. 검증 손실(괄호 안 첫 숫자)은 epoch 0 3.90 → epoch 7 1.44.
- 마지막 번역: `Ein Mann balanciert einen <unk> auf seinem Arm .` → **`A man is balancing a <unk> on his arm .`**
