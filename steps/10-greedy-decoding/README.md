# 한 단어씩 생성하기: greedy decoding

- 원본: [추론](https://dev-jonghoonpark.github.io/annotated-transformer-modern/#inference) · [탐욕적 디코딩](https://dev-jonghoonpark.github.io/annotated-transformer-modern/#greedy-decoding)

모델이 완성됐으니 문장을 만들어 보자. 디코더는 한 번에 한 토큰만 정한다.

1. 원문을 한 번만 `encode`해서 `memory`를 만든다.
2. 시작 토큰 하나(`start_symbol`)로 `ys`를 시작한다.
3. `ys`를 `decode`하고 **마지막 위치**의 확률 분포(`out[:, -1]`)에서 가장 높은 토큰을 고른다.
4. 그 토큰을 `ys` 뒤에 붙이고 3번으로 돌아간다.

매번 가장 확률이 높은 것 하나만 고르므로 "greedy(탐욕적)"라고 부른다. 논문은 beam search를 썼지만 원본 노트북은 단순하게 greedy만 구현했다.

> 원본 노트북에서는 `inference_test` 안에 같은 반복문이 들어 있고 `greedy_decode`는 2부에 따로 나온다. 여기서는 같은 코드를 `greedy_decode` 함수 하나로 합쳐 먼저 소개했다.

## 실행 결과에서 볼 것

원본의 `inference_test`를 10번 실행한다. 학습하지 않은 모델이라 입력 `1..10`과 상관없이 같은 토큰을 반복하거나 둘을 번갈아 낸다. 이제 이 모델을 학습시킬 차례다.
