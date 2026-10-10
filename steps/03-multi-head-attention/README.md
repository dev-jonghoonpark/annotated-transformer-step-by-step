# 머리를 여러 개로: Multi-Head Attention

- 원본: [어텐션](https://dev-jonghoonpark.github.io/annotated-transformer-modern/#attention) (Multi-head 부분) · [모델에서 어텐션을 쓰는 곳](https://dev-jonghoonpark.github.io/annotated-transformer-modern/#applications-of-attention-in-our-model)

![Multi-Head Attention](images/ModalNet-20.png)

어텐션 한 번은 토큰마다 "가중 평균 하나"만 만든다. 그런데 한 단어가 동시에 신경 써야 할 관계는 여러 가지다(주어-동사, 수식 관계, 바로 옆 단어...). 그래서 512차원을 8조각(64차원씩)으로 나눠 **어텐션 8개를 따로 돌리고** 결과를 이어 붙인다.

$$\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h})W^O,\quad \mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)$$

이번 단계에서 추가되는 것:

- **`clones(module, N)`**: 같은 모듈을 N개 deepcopy한다. 모양은 같지만 가중치는 따로 갖는다. 앞으로 층을 쌓을 때마다 쓴다.
- **`MultiHeadedAttention`**: `nn.Linear` 4개(W_Q, W_K, W_V, W_O)를 갖는다. head 8개를 반복문으로 돌리지 않고, **`view` + `transpose`로 head를 배치 차원처럼 앞으로 빼서** 한 번의 행렬곱으로 처리하는 게 핵심이다. 계산한 가중치는 `self.attn`에 남겨 둔다.
- **`attention`에 `dropout` 인자 추가**: 학습 때 가중치 일부를 0으로 만들어 과적합을 줄인다.

## 실행 결과에서 볼 것

- 입력과 출력의 모양이 같다. 그래서 이 블록을 몇 층이든 쌓을 수 있다.
- `(2, 10, 512)` → `(2, 10, 8, 64)` → `(2, 8, 10, 64)`로 모양이 바뀌는 과정.
- 같은 가중치를 넣었을 때 PyTorch 내장 `nn.MultiheadAttention`과 결과 차이가 1e-7 수준이다. 우리가 만든 것이 표준 구현과 같은 계산이라는 뜻이다.
- W_O 실험: 이어 붙인 head 출력에 W_O를 곱한 값이 `ours(y, y, y)`와 정확히 같고, head마다 W_O의 자기 몫을 곱해 더한 값과도 같다(차이 1e-7 수준).

## 참고

### W_O(`self.linears[-1]`)의 역할

head 8개의 결과(64차원씩)를 이어 붙이면 다시 512차원이 되지만, 이 상태에서는 0~63번 칸에 head 1 결과만, 64~127번 칸에 head 2 결과만 들어 있어 head끼리 정보가 섞이지 않는다. 마지막 `nn.Linear(512, 512)`가 이 512칸 전체를 섞어서 출력 한 칸 한 칸이 모든 head의 결과를 반영하게 만든다. 수식으로는 $\mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h})W^O = \sum_i \mathrm{head_i} W^O_i$, 즉 head마다 W_O의 자기 몫(64행)을 곱해 **더하는** 것과 같다. 출력 크기를 다시 d_model로 맞춰 다음 층에 넘기는 역할도 한다.

### `forward`가 하는 일

`forward`는 `mha(q, k, v)`처럼 모듈을 함수처럼 부를 때 실행되는 본체다. 코드 주석의 1) 2) 3) 순서로 읽으면 된다. 모양은 데모 기준(문장 2개, 토큰 10개, d_model 512, head 8개)이다.

- **0) 마스크에 head 축 추가** `mask.unsqueeze(1)`: `(2, 1, 10)` → `(2, 1, 1, 10)`. 크기 1인 축은 브로드캐스트되므로 head 8개에 같은 마스크가 걸린다.
- **1) head로 나누기**: W_Q·W_K·W_V(`linears[0..2]`)로 투영한 `(2, 10, 512)`를 `.view(2, -1, 8, 64)`로 head 8개 × 64칸으로 자르고 `.transpose(1, 2)`로 `(2, 8, 10, 64)`를 만든다. head가 앞으로 나와 "표를 쌓은 장 수"가 되므로, 1단계 `attention`이 head 8개를 한 번에 계산한다.
- **2) 어텐션**: 출력 `(2, 8, 10, 64)`, 가중치 `self.attn` `(2, 8, 10, 10)`.
- **3) 합치기**: `.transpose(1, 2)`로 `(2, 10, 8, 64)`로 되돌리고 `.view(2, -1, 512)`로 head를 옆으로 이어 붙인(Concat) 뒤 W_O(`linears[-1]`)를 통과시킨다. 중간의 `.contiguous()`는 transpose로 흐트러진 메모리 배치를 다시 정리하는 것이다. 이게 없으면 `view`가 에러를 낸다.

마지막의 `del query, key, value`는 중간 텐서를 일찍 놓아 메모리를 아끼려는 줄이고, 결과에는 영향이 없다.
