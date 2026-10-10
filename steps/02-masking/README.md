# 보면 안 되는 칸 가리기

- 원본: [디코더](https://dev-jonghoonpark.github.io/annotated-transformer-modern/#decoder) · [어텐션](https://dev-jonghoonpark.github.io/annotated-transformer-modern/#attention)

어텐션은 기본적으로 모든 토큰을 본다. 하지만 보면 안 되는 칸이 두 종류 있다.

1. **미래 토큰**: 번역문을 한 단어씩 만들 때, 학습 중에도 아직 만들지 않은 뒤쪽 단어를 보면 반칙이다.
2. **패딩**: 길이가 다른 문장을 한 배치로 묶으려고 채워 넣은 `<blank>` 자리는 의미가 없다.

그래서 `attention`에 `mask` 인자가 생긴다(diff의 빨간 줄 → 초록 줄). 가릴 칸의 점수를 softmax 전에 `-1e9`로 덮으면 softmax 뒤 그 칸의 가중치는 사실상 0이 된다.

`subsequent_mask(size)`는 미래를 가리는 마스크다. `torch.triu(..., diagonal=1)`로 대각선 위쪽을 1로 만든 뒤 `== 0`으로 뒤집어서, **i번째 행은 0..i번째 열만 True**인 아래 삼각 행렬을 만든다. 맨 앞의 1차원은 나중에 배치 차원에 맞춰 브로드캐스트하기 위한 것이다.

## 실행 결과에서 볼 것

- 마스크를 건 가중치 행렬은 대각선 위가 전부 0이다.
- 패딩 마스크를 걸면 패딩 열의 가중치가 0이 된다.
- 차트는 원본 노트북의 `example_mask`다. 행(Masking)이 볼 수 있는 열(Window)만 노랗다.
