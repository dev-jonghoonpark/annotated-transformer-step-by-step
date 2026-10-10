# 깊게 쌓아도 무너지지 않게: LayerNorm과 잔차 연결

- 원본: [인코더](https://dev-jonghoonpark.github.io/annotated-transformer-modern/#encoder)

이제 블록을 6층씩 쌓을 준비를 한다. 층이 깊어지면 신호가 점점 작아지거나 커져서 학습이 망가진다. 이를 막는 장치가 두 개 있다.

- **`LayerNorm`**: 토큰 벡터 하나(512차원)를 평균 0, 표준편차 1로 맞춘 뒤 학습 가능한 스케일 `a_2`와 이동 `b_2`를 적용한다.
- **`SublayerConnection`**: 서브층(어텐션이나 FFN)을 감싸서 `x + dropout(sublayer(norm(x)))`를 돌려준다. 서브층의 결과를 입력에 **더하기** 때문에 서브층이 아무것도 못 해도 입력은 그대로 통과한다.

> 논문 수식은 `LayerNorm(x + Sublayer(x))`(Post-LN)인데 코드는 정규화를 먼저 한다(Pre-LN). 원본 코드 주석에 "for code simplicity the norm is first"라고 적혀 있다. Pre-LN은 학습이 더 안정적이어서 요즘 LLM도 대부분 이 순서를 쓴다.

`sublayer`를 **함수로 받는다**는 점도 눈여겨보자. 다음 단계에서 어텐션은 `lambda x: self.self_attn(x, x, x, mask)` 형태로, FFN은 모듈 그대로 넘긴다.

## 실행 결과에서 볼 것

- 정규화 뒤 평균 0, 표준편차 1.
- 서브층이 0만 내놓아도 출력이 입력과 같다.
- FFN 20개를 그냥 쌓으면 신호가 0.05 수준으로 쪼그라들지만, 잔차 연결과 정규화로 감싸면 크기가 유지된다.
