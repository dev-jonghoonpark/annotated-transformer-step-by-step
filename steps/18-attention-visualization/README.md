# 학습된 모델 들여다보기: 어텐션 시각화

- 원본: [어텐션 시각화](https://dev-jonghoonpark.github.io/annotated-transformer-modern/#attention-visualization)

1단계에서 `attention`이 가중치 `p_attn`을 돌려주게 했고, 3단계에서 `MultiHeadedAttention`이 그것을 `self.attn`에 저장해 두게 했다. 그 가중치를 꺼내 그림으로 그린다.

- **`get_encoder`/`get_decoder_self`/`get_decoder_src`**: 층 번호를 받아 그 층의 `attn`(마지막 forward 때의 가중치)을 꺼낸다.
- **`mtx2df`, `attn_map`**: 가중치 행렬을 DataFrame으로 바꿔 altair 히트맵으로 그린다. 행은 query 토큰, 열은 key 토큰이다.
- **`visualize_layer`**: 한 층의 head 8개 중 짝수 번째 4개(0, 2, 4, 6)를 가로로 붙인다.

차트 위에 마우스를 올리면 칸마다 토큰과 값이 보인다. 차트는 드래그·휠로 움직일 수 있다.

> 원본은 디코더 셀프 어텐션 그림의 행·열 이름에도 원문(독일어) 토큰을 붙인다. 원본 그대로 두었으므로 그 그림의 축 이름은 실제 토큰과 맞지 않을 수 있다.

## 실행 결과에서 볼 것

- **인코더 셀프 어텐션**(1, 3, 5층): 얕은 층에서는 자기 자신이나 바로 옆 토큰을 보는 head가 많다.
- **디코더 소스 어텐션**: 영어 단어를 만들 때 대응하는 독일어 단어 쪽이 밝다. `Sportkleidung` ↔ `athletic clothes`처럼 번역 대응이 보인다.
