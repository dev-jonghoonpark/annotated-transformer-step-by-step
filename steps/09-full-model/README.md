# 하나로 조립: EncoderDecoder와 make_model

- 원본: [모델 구조](https://dev-jonghoonpark.github.io/annotated-transformer-modern/#model-architecture) · [전체 모델](https://dev-jonghoonpark.github.io/annotated-transformer-modern/#full-model)

원본 노트북이 **맨 처음에** 보여 주는 코드가 이 단계의 `EncoderDecoder`다. 원본을 위에서부터 읽으면 아직 아무 부품도 모르는 상태에서 이 클래스를 만나서 어디부터 봐야 할지 막막하다. 여기서는 부품을 다 만든 뒤에 만나므로 각 줄이 무엇을 부르는지 이미 알고 있다.

- **`EncoderDecoder`**: 부품 다섯 개(encoder, decoder, src_embed, tgt_embed, generator)를 들고 있는 껍데기다. `encode`로 원문을 `memory`로 만들고, `decode`로 memory와 지금까지의 번역문을 디코더에 넣는다.
- **`Generator`**: 디코더 출력(512차원)을 사전 크기로 투영하고 `log_softmax`를 걸어 "다음 토큰이 무엇일지"의 로그 확률을 낸다.
- **`make_model`**: 하이퍼파라미터(N=6, d_model=512, d_ff=2048, h=8, dropout=0.1)로 모델을 조립한다. 어텐션·FFN·PE를 하나씩 만들어 두고 `copy.deepcopy`로 필요한 곳마다 복제한다. 마지막에 2차원 이상인 파라미터를 전부 Xavier(Glorot) 균등분포로 초기화한다. 원본 주석대로 "This was important"다.

## 실행 결과에서 볼 것

- 15단계에서 만들 사전 크기(독일어 8,316, 영어 6,384)로 조립하면 파라미터가 약 5,500만 개다. 디코더가 가장 크다.
- `encode` → `decode` → `generator`를 지나며 모양이 `(1, 9)` → `(1, 9, 512)` → `(1, 5, 512)` → `(1, 5, 6384)`로 바뀐다.
- 초기화된 가중치의 최댓값이 Xavier 이론 한계 $\sqrt{6/(512+2048)}$와 같다.
