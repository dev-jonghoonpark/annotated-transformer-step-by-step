# 단어를 벡터로, 순서를 신호로

- 원본: [임베딩과 소프트맥스](https://dev-jonghoonpark.github.io/annotated-transformer-modern/#embeddings-and-softmax) · [위치 인코딩](https://dev-jonghoonpark.github.io/annotated-transformer-modern/#positional-encoding)

모델에 들어가는 것은 단어 id(정수)다. 이것을 512차원 벡터로 바꾸는 부품 두 개를 만든다.

**`Embeddings`**: `nn.Embedding` 표에서 id에 해당하는 행을 꺼낸 뒤 $\sqrt{d_{model}}$을 곱한다. 바로 뒤에 더할 위치 인코딩 값(-1~1)과 크기를 맞추기 위해서다.

**`PositionalEncoding`**: 어텐션은 순서를 모른다. 단어 순서를 섞어 넣어도 결과가 그대로 섞여 나올 뿐이다. 그래서 위치마다 고유한 사인·코사인 무늬를 임베딩에 **더한다**.

$$PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{model}}),\quad PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

차원마다 파장이 다르다. 앞쪽 차원은 빠르게, 뒤쪽 차원은 느리게 진동한다. 코드는 `div_term`을 로그 공간에서 계산하고, 짝수 열에 sin, 홀수 열에 cos을 채운다. 학습하는 값이 아니라서 `nn.Parameter`가 아닌 `register_buffer`로 등록한다.

## 실행 결과에서 볼 것

- `× √512`를 하면 임베딩 표준편차가 약 1에서 약 22로 커진다.
- 위치 0은 `[0, 1, 0, 1, ...]`(sin 0, cos 0)로 시작한다.
- 같은 단어(id 42)가 두 위치에 있으면 임베딩은 같지만 PE를 더한 뒤에는 달라진다.
- 차트는 원본의 `example_positional`이다. 차원 4~7의 파형인데, 차원 번호가 클수록 주기가 길다.
