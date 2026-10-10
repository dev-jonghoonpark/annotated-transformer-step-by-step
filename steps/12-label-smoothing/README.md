# 손실 함수: 너무 확신하지 않게

- 원본: [정규화 → 레이블 스무딩](https://dev-jonghoonpark.github.io/annotated-transformer-modern/#label-smoothing) · [손실 계산](https://dev-jonghoonpark.github.io/annotated-transformer-modern/#loss-computation)

정답 토큰에 확률 1을 주는 one-hot 대신, **정답에 0.9, 나머지에 0.1을 나눠 준** 분포를 목표로 학습한다. 논문은 이를 label smoothing($\epsilon_{ls}=0.1$)이라고 부른다. 모델이 지나치게 확신하지 않게 되어 perplexity는 조금 나빠지지만 정확도와 BLEU는 좋아진다.

**`LabelSmoothing`**: 목표 분포 `true_dist`를 만든 뒤 모델의 로그 확률과의 KL divergence를 손실로 쓴다.
- 모든 칸을 `smoothing / (size - 2)`로 채운다. 2를 빼는 이유는 정답 칸과 패딩 칸을 제외하기 때문이다.
- 정답 칸(`scatter_`)에 `confidence = 1 - smoothing`을 넣는다.
- 패딩 열은 0으로 만들고, 정답 자체가 패딩인 행(문장 뒤 채움)은 통째로 0으로 만들어 손실에서 뺀다.

**`SimpleLossCompute`**: generator를 거쳐 손실을 계산하고 토큰 수로 나눈다. 두 값을 돌려준다. 하나는 기록용 합계(`sloss.data * norm`)이고, 하나는 `backward()`를 부를 텐서(`sloss`)다.

## 실행 결과에서 볼 것

- 원본의 `example_label_smoothing`: 정답 칸 0.6, 나머지 0.133씩, 패딩이 정답인 3번째 행은 전부 0이다.
- 원본의 `penalization_visualization`: 정답 확률을 계속 키우면 손실이 내려가다가 **다시 올라간다**. 지나친 확신에 벌점을 준다는 뜻이다.
