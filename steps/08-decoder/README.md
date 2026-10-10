# 디코더: 원문을 보면서, 미래는 보지 않고

- 원본: [디코더](https://dev-jonghoonpark.github.io/annotated-transformer-modern/#decoder)

그림 오른쪽이 디코더다. 인코더 층과 다른 점은 서브층이 하나 더 있다는 것이다.

1. **masked self-attention**: 지금까지 만든 번역 토큰끼리 본다. `tgt_mask`로 미래를 가린다.
2. **source attention**(encoder-decoder attention): query는 디코더에서, key와 value는 인코더 출력 `memory`에서 가져온다. "다음 단어를 고르려면 원문의 어디를 봐야 하나?"를 계산하는 자리다. 코드에서 `self.src_attn(x, m, m, src_mask)`다.
3. **FFN**

`Decoder`는 `Encoder`와 똑같이 N개를 쌓고 마지막에 LayerNorm을 건다. 차이는 `memory`, `src_mask`, `tgt_mask`를 함께 넘긴다는 것뿐이다.

## 실행 결과에서 볼 것

- 디코더 층은 서브층이 3개라 인코더보다 파라미터가 많다(약 2,520만 개).
- 4, 5번째 입력 토큰만 바꾸면 1~3번째 출력은 **차이가 정확히 0**이다. 미래를 볼 수 없다는 뜻이다.
- 마스크를 빼면 1번째 출력까지 바뀐다. 마스크가 없으면 모델이 뒤에 올 정답을 엿보고 학습하게 된다.
