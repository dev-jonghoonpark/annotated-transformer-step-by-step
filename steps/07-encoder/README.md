# 인코더: 같은 층을 6번

- 원본: [인코더와 디코더 스택](https://dev-jonghoonpark.github.io/annotated-transformer-modern/#encoder-and-decoder-stacks)

![Transformer 구조](images/ModalNet-21.png)

그림 왼쪽이 인코더다. 지금까지 만든 부품을 조립한다.

- **`EncoderLayer`**: self-attention과 FFN을 각각 `SublayerConnection`으로 감싼 두 단계다. self-attention이므로 query, key, value에 모두 같은 `x`를 넣는다.
- **`Encoder`**: `EncoderLayer`를 `clones`로 N개 복제해 차례로 통과시키고, 마지막에 LayerNorm을 한 번 더 건다. Pre-LN 구조에서는 마지막 층의 출력이 정규화되지 않은 상태라 이 정규화가 필요하다.

`EncoderLayer`는 어텐션과 FFN 객체를 **생성자 인자로 받는다**. 어떤 어텐션을 쓸지는 조립하는 쪽(9단계의 `make_model`)이 정한다.

## 실행 결과에서 볼 것

- 6층 인코더의 파라미터는 약 1,890만 개다.
- 층마다 가중치가 다른 객체다(`clones`가 deepcopy하기 때문).
- 패딩 자리 값을 100배 크기의 난수로 바꿔도 진짜 토큰 4개의 출력은 **완전히 같다**(차이 0). 2단계의 마스크가 6층을 통과하는 동안 계속 지켜진다.
