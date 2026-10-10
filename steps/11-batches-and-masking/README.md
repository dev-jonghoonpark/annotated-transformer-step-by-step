# 학습용 배치: 한 칸 밀기와 마스크

- 원본: [배치와 마스킹](https://dev-jonghoonpark.github.io/annotated-transformer-modern/#batches-and-masking)

여기서부터 2부, 학습이다. 첫 부품은 문장쌍 배치를 학습에 필요한 형태로 바꾸는 `Batch` 클래스다.

**teacher forcing**: 학습할 때는 디코더에 정답 문장을 통째로 넣고, 모든 위치에서 동시에 "다음 토큰"을 맞히게 한다. 그러려면 정답을 한 칸 밀어야 한다.

- `tgt` = 정답에서 마지막 토큰을 뺀 것 → 디코더 입력
- `tgt_y` = 정답에서 첫 토큰을 뺀 것 → 맞혀야 할 답

i번째 위치는 `tgt[0..i]`를 보고 `tgt_y[i]`를 맞힌다. 이때 i번째 위치가 `tgt[i+1..]`를 보면 답을 베끼는 셈이므로 **미래 마스크**가 필요하다. `make_std_mask`는 패딩 마스크와 `subsequent_mask`를 AND로 합친다.

`ntokens`는 패딩이 아닌 정답 토큰 수다. 손실을 이 값으로 나눠 토큰당 평균 손실을 만든다.

## 실행 결과에서 볼 것

- `tgt`와 `tgt_y`가 한 칸 어긋나 있다.
- `tgt_mask`는 아래 삼각 행렬인데, 패딩 열(마지막 열)이 추가로 0이다.
