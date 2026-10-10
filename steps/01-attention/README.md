# 어텐션 한 줄부터

- 원본: [어텐션](https://dev-jonghoonpark.github.io/annotated-transformer-modern/#attention)

Transformer를 한 문장으로 줄이면 "모든 토큰이 다른 모든 토큰을 얼마나 참고할지 스스로 정해서 섞는 모델"이다. 그 "섞는" 연산이 어텐션이고, 나머지 부품은 전부 이 연산을 잘 돌리기 위한 장치다. 그래서 첫 줄은 어텐션부터 쓴다.

![Scaled Dot-Product Attention](images/ModalNet-19.png)

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **query와 key의 내적** `QK^T`: "i번째 토큰이 j번째 토큰과 얼마나 관련 있는가"를 점수로 매긴다.
- **softmax**: 점수를 행마다 합이 1인 가중치로 바꾼다.
- **value에 곱하기**: 그 가중치로 value 벡터들을 섞는다. 출력 토큰 하나는 입력 토큰들의 가중 평균이다.
- **`/ math.sqrt(d_k)`**: 벡터 차원이 커질수록 내적값이 커져서 softmax가 한 칸에 쏠린다. 차원 수의 제곱근으로 나눠 이를 막는다.

함수가 결과와 함께 가중치 `p_attn`도 돌려주는 점을 기억해 두자. 마지막 단계에서 이 가중치를 그림으로 그린다.

## 실행 결과에서 볼 것

- 가중치 행렬의 각 행을 더하면 1이다.
- 출력 첫 번째 토큰을 가중치로 직접 섞어 계산한 값과 비교하면 똑같다.
- 나누지 않으면 `d_k=512`에서 softmax가 한두 칸에 거의 전부를 몰아준다. 나누면 고르게 퍼진다.

## 참고: `key.transpose(-2, -1)`의 `-2`, `-1`

`transpose(dim0, dim1)`는 텐서의 두 차원을 맞바꾼다. 인자는 차원 번호이고, 음수는 뒤에서부터 센다. `-1`은 마지막 차원, `-2`는 그 앞 차원이다.

이 단계의 key는 `(batch, 토큰 수, d_k)` = `(1, 3, 4)` 모양이다.

| 차원 번호 | 0 | 1 | 2 |
| --- | --- | --- | --- |
| 음수로 쓰면 | -3 | -2 | -1 |
| 의미 | batch | 토큰 수 | d_k |

`key.transpose(-2, -1)`은 토큰 수와 d_k 자리를 바꿔 `(1, 4, 3)`을 만든다. 수식의 $K^T$다.

- `torch.matmul`은 **마지막 두 차원끼리** 행렬곱을 하고, 그 앞 차원(batch)은 그대로 둔다.
- 그래서 `query (1, 3, 4) @ keyᵀ (1, 4, 3)` → `scores (1, 3, 3)`이 된다. `scores[0, i, j]`가 "i번째 query와 j번째 key의 내적"이다.
- transpose 없이 `(1, 3, 4) @ (1, 3, 4)`를 하면 안쪽 차원(4와 3)이 맞지 않아 에러가 난다.

**왜 `transpose(1, 2)`가 아니라 음수로 쓸까?** 3단계 Multi-Head Attention에서 key는 `(batch, head, 토큰 수, d_k)`, 4차원이 된다. 이때 `transpose(1, 2)`라고 쓰면 head와 토큰 수가 바뀌어 틀린 계산이 된다. `-2, -1`은 앞에 차원이 몇 개 붙든 항상 "마지막 두 차원"을 가리키므로 같은 `attention` 함수를 3차원에도 4차원에도 그대로 쓸 수 있다. 참고로 최신 PyTorch에서는 같은 일을 하는 `key.mT`도 있다.
