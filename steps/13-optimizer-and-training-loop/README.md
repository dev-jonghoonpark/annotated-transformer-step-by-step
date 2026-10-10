# 학습률 스케줄과 학습 루프

- 원본: [학습 루프](https://dev-jonghoonpark.github.io/annotated-transformer-modern/#training-loop) · [옵티마이저](https://dev-jonghoonpark.github.io/annotated-transformer-modern/#optimizer)

**`rate`**: 논문의 학습률 공식이다. 원본 노트북이 "This part is very important"라고 강조한 부분이다.

$$lrate = d_{model}^{-0.5} \cdot \min(step^{-0.5},\ step \cdot warmup^{-1.5})$$

처음 `warmup` 스텝 동안은 학습률을 직선으로 올리고, 그 뒤로는 $1/\sqrt{step}$에 비례해 내린다. 초반에 큰 학습률로 시작하면 Adam의 통계가 덜 쌓인 상태에서 크게 튀기 때문에 천천히 올린다. step 0에서는 0의 음수 제곱이 되므로 1로 바꿔 둔다. PyTorch의 `LambdaLR`에 넘기면 `optimizer`의 기본 학습률(`lr=1`)에 이 값이 곱해진다.

**`run_epoch`**: 데이터를 한 바퀴 돈다. 배치마다 `forward` → 손실 → `backward` → (`accum_iter`번에 한 번) `optimizer.step()` → `scheduler.step()` 순서다. 40배치마다 손실과 처리 속도를 출력한다. `mode="eval"`이면 역전파 없이 손실만 잰다.

**`TrainState`**, **`DummyOptimizer`/`DummyScheduler`**: 진행 상황을 세는 상자와, 평가할 때 `run_epoch`에 넘길 "아무것도 안 하는" 옵티마이저다.

## 실행 결과에서 볼 것

원본의 `example_learning_schedule` 차트다. 모델 크기와 warmup에 따라 꼭짓점의 위치와 높이가 달라진다. d_model이 작을수록 학습률이 높다. (원본은 2만 스텝을 모두 그리지만 페이지 크기를 줄이려고 20스텝마다 하나씩만 그렸다.)
