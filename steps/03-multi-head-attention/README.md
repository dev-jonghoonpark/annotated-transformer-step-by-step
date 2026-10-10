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

### `forward`가 하는 일

`forward`는 모듈을 함수처럼 부를 때 실행되는 본체다. 데모에서 `mha = MultiHeadedAttention(h=8, d_model=512)`(mha = Multi-Head Attention)로 만든 뒤 `mha(x, x, x)`라고 부르면 이 함수가 돈다. 코드 주석의 1) 2) 3) 순서로 읽으면 된다. 모양은 데모 기준(문장 2개, 토큰 10개, d_model 512, head 8개)이다.

- **0) 마스크에 head 축 추가** `mask.unsqueeze(1)`: `(2, 1, 10)` → `(2, 1, 1, 10)`. 크기 1인 축은 브로드캐스트되므로 head 8개에 같은 마스크가 걸린다.
- **1) head로 나누기**: W_Q·W_K·W_V(`linears[0..2]`)로 투영한 `(2, 10, 512)`를 `.view(2, -1, 8, 64)`로 head 8개 × 64칸으로 자르고 `.transpose(1, 2)`로 `(2, 8, 10, 64)`를 만든다. head가 앞으로 나와 "표를 쌓은 장 수"가 되므로, 1단계 `attention`이 head 8개를 한 번에 계산한다.
- **2) 어텐션**: 출력 `(2, 8, 10, 64)`, 가중치 `self.attn` `(2, 8, 10, 10)`.
- **3) 합치기**: `.transpose(1, 2)`로 `(2, 10, 8, 64)`로 되돌리고 `.view(2, -1, 512)`로 head를 옆으로 이어 붙인(Concat) 뒤 W_O(`linears[-1]`)를 통과시킨다. 중간의 `.contiguous()`는 transpose로 흐트러진 메모리 배치를 다시 정리하는 것이다. 이게 없으면 `view`가 에러를 낸다.

마지막의 `del query, key, value`는 중간 텐서를 일찍 놓아 메모리를 아끼려는 줄이고, 결과에는 영향이 없다.

### W_O(`self.linears[-1]`)의 역할

head 8개의 결과(64차원씩)를 이어 붙이면 다시 512차원이 되지만, 이 상태에서는 0~63번 칸에 head 1 결과만, 64~127번 칸에 head 2 결과만 들어 있어 head끼리 정보가 섞이지 않는다. 마지막 `nn.Linear(512, 512)`가 이 512칸 전체를 섞어서 출력 한 칸 한 칸이 모든 head의 결과를 반영하게 만든다. 수식으로는 $\mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h})W^O = \sum_i \mathrm{head_i} W^O_i$, 즉 head마다 W_O의 자기 몫(64행)을 곱해 **더하는** 것과 같다. 출력 크기를 다시 d_model로 맞춰 다음 층에 넘기는 역할도 한다.

### head 축을 앞으로 빼는 이유 (`transpose(1, 2)`)

`attention`은 마지막 두 축을 "표 한 장"으로 보고 계산한다. view 직후 `(2, 10, 8, 64)`는 마지막 두 축이 (head, d_k)라서, 그대로 넣으면 토큰끼리가 아니라 head끼리 비교한 8×8 표가 나온다. head 8은 비교 대상이 아니라 "같은 계산을 따로 몇 번 하느냐"이므로 문장 수처럼 앞으로 빼서 묶음 축으로 둔다.

<figure class="fig"><svg viewBox="0 0 560 250" role="img" aria-label="view 직후와 transpose 후의 축 배치"><defs><marker id="arA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" class="arrowhead"/></marker></defs><text class="label mono" x="20" y="18">view 직후 (2, 10, 8, 64)</text><rect class="box" x="20" y="28" width="120" height="46" rx="6"/><text class="num" x="80.0" y="49" text-anchor="middle">2</text><text class="muted" x="80.0" y="66" text-anchor="middle">문장</text><rect class="box" x="150" y="28" width="120" height="46" rx="6"/><text class="num" x="210.0" y="49" text-anchor="middle">10</text><text class="muted" x="210.0" y="66" text-anchor="middle">토큰</text><rect class="bad" x="280" y="28" width="120" height="46" rx="6"/><text class="num" x="340.0" y="49" text-anchor="middle">8</text><text class="muted" x="340.0" y="66" text-anchor="middle">head</text><rect class="bad" x="410" y="28" width="120" height="46" rx="6"/><text class="num" x="470.0" y="49" text-anchor="middle">64</text><text class="muted" x="470.0" y="66" text-anchor="middle">d_k</text><path class="line" d="M280 80 v6 H530 v-6"/><text class="t-bad" x="405" y="104" text-anchor="middle">마지막 두 축 = (head, d_k) → head끼리 비교 ✗</text><path class="line" d="M210 114 L340 150" marker-end="url(#arA)"/><path class="line" d="M340 114 L210 150" marker-end="url(#arA)"/><text class="label mono" x="380" y="138">transpose(1, 2)</text><text class="label mono" x="20" y="138">(2, 8, 10, 64)</text><rect class="grp" x="20" y="158" width="120" height="46" rx="6"/><text class="num" x="80.0" y="179" text-anchor="middle">2</text><text class="muted" x="80.0" y="196" text-anchor="middle">문장</text><rect class="grp" x="150" y="158" width="120" height="46" rx="6"/><text class="num" x="210.0" y="179" text-anchor="middle">8</text><text class="muted" x="210.0" y="196" text-anchor="middle">head</text><rect class="tbl" x="280" y="158" width="120" height="46" rx="6"/><text class="num" x="340.0" y="179" text-anchor="middle">10</text><text class="muted" x="340.0" y="196" text-anchor="middle">토큰</text><rect class="tbl" x="410" y="158" width="120" height="46" rx="6"/><text class="num" x="470.0" y="179" text-anchor="middle">64</text><text class="muted" x="470.0" y="196" text-anchor="middle">d_k</text><path class="line" d="M20 210 v6 H270 v-6"/><path class="line" d="M280 210 v6 H530 v-6"/><text class="label" x="145" y="234" text-anchor="middle">묶음: 2 × 8 = 16장</text><text class="t-good" x="405" y="234" text-anchor="middle">표 한 장 = (토큰, d_k) ✓</text></svg></figure>

### 점수표 `(2, 8, 10, 10)`의 의미

`attention` 안의 `query @ key.transpose(-2, -1)` 결과다. 문장 2개 × head 8개 = 16장의 10×10 "토큰 ↔ 토큰" 점수표로, 행은 query 토큰, 열은 key 토큰이다. 데모 출력의 `head별 가중치 (2, 8, 10, 10)`은 여기에 softmax를 건 것이다.

<figure class="fig"><svg viewBox="0 0 560 262" role="img" aria-label="10×10 점수표 16장이 쌓인 모습"><rect class="box" x="94" y="38" width="140" height="140"/><rect class="box" x="86" y="46" width="140" height="140"/><rect class="box" x="78" y="54" width="140" height="140"/><rect class="tbl" x="70" y="62" width="140" height="140"/><line class="grid" x1="84" y1="62" x2="84" y2="202"/><line class="grid" x1="70" y1="76" x2="210" y2="76"/><line class="grid" x1="98" y1="62" x2="98" y2="202"/><line class="grid" x1="70" y1="90" x2="210" y2="90"/><line class="grid" x1="112" y1="62" x2="112" y2="202"/><line class="grid" x1="70" y1="104" x2="210" y2="104"/><line class="grid" x1="126" y1="62" x2="126" y2="202"/><line class="grid" x1="70" y1="118" x2="210" y2="118"/><line class="grid" x1="140" y1="62" x2="140" y2="202"/><line class="grid" x1="70" y1="132" x2="210" y2="132"/><line class="grid" x1="154" y1="62" x2="154" y2="202"/><line class="grid" x1="70" y1="146" x2="210" y2="146"/><line class="grid" x1="168" y1="62" x2="168" y2="202"/><line class="grid" x1="70" y1="160" x2="210" y2="160"/><line class="grid" x1="182" y1="62" x2="182" y2="202"/><line class="grid" x1="70" y1="174" x2="210" y2="174"/><line class="grid" x1="196" y1="62" x2="196" y2="202"/><line class="grid" x1="70" y1="188" x2="210" y2="188"/><rect class="hit" x="140" y="90" width="14" height="14"/><text class="muted" x="140" y="222" text-anchor="middle">key 토큰 10 →</text><text class="muted" transform="translate(58 132) rotate(-90)" text-anchor="middle">query 토큰 10 →</text><text class="muted" x="244" y="42">… 16장</text><text class="label" x="70" y="242">맨 앞 = 문장 0 · head 0</text><text class="label mono" x="300" y="40">query (2, 8, 10, 64)</text><text class="label mono" x="300" y="67">@ keyᵀ (2, 8, 64, 10)</text><text class="label mono strong" x="300" y="94">= scores (2, 8, 10, 10)</text><text class="label " x="300" y="131">앞 (2, 8): 문장 2 × head 8 = 16장</text><text class="label " x="300" y="158">뒤 (10, 10): 표 한 장</text><text class="label " x="300" y="185">칠한 칸 (2, 5): 토큰 2가 토큰 5를</text><text class="label " x="300" y="212">얼마나 참고할지의 점수</text></svg></figure>

### 마스크에 `unsqueeze(1)`을 하는 이유

PyTorch는 모양이 다른 텐서를 **오른쪽 끝부터** 맞춘다(브로드캐스트). 3차원 마스크 `(2, 1, 10)`을 그대로 쓰면 마스크의 문장 축이 점수표의 head 축에 붙는다. 문장 수와 head 수가 다르면 에러가 나고, 같으면(둘 다 8) 에러 없이 엉뚱한 칸이 가려진다. `unsqueeze(1)`로 head 자리에 크기 1인 축을 끼우면 문장 축끼리 맞고, 모든 head에 같은 마스크가 퍼진다. 원본 주석 "Same mask applied to all h heads"가 이 뜻이다.

<figure class="fig"><svg viewBox="0 0 660 250" role="img" aria-label="마스크를 오른쪽 끝부터 맞추는 브로드캐스트"><defs><marker id="arC" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" class="arrowhead"/></marker></defs><text class="muted" x="233" y="20" text-anchor="middle">문장</text><text class="muted" x="307" y="20" text-anchor="middle">head</text><text class="muted" x="381" y="20" text-anchor="middle">query</text><text class="muted" x="455" y="20" text-anchor="middle">key</text><text class="label mono" x="20" y="52">scores</text><rect class="box" x="200" y="28" width="66" height="36" rx="5"/><text class="num" x="233" y="52" text-anchor="middle">2</text><rect class="box" x="274" y="28" width="66" height="36" rx="5"/><text class="num" x="307" y="52" text-anchor="middle">8</text><rect class="box" x="348" y="28" width="66" height="36" rx="5"/><text class="num" x="381" y="52" text-anchor="middle">10</text><rect class="box" x="422" y="28" width="66" height="36" rx="5"/><text class="num" x="455" y="52" text-anchor="middle">10</text><text class="label mono" x="20" y="116">mask 그대로</text><rect class="bad" x="274" y="92" width="66" height="36" rx="5"/><text class="num" x="307" y="116" text-anchor="middle">2</text><rect class="box" x="348" y="92" width="66" height="36" rx="5"/><text class="num" x="381" y="116" text-anchor="middle">1</text><rect class="box" x="422" y="92" width="66" height="36" rx="5"/><text class="num" x="455" y="116" text-anchor="middle">10</text><text class="t-bad" x="500" y="116">✗ 문장 축이 head 자리에</text><text class="label mono" x="20" y="180">mask.unsqueeze(1)</text><rect class="good" x="200" y="156" width="66" height="36" rx="5"/><text class="num" x="233" y="180" text-anchor="middle">2</text><rect class="grp" x="274" y="156" width="66" height="36" rx="5"/><text class="num" x="307" y="180" text-anchor="middle">1</text><rect class="box" x="348" y="156" width="66" height="36" rx="5"/><text class="num" x="381" y="180" text-anchor="middle">1</text><rect class="box" x="422" y="156" width="66" height="36" rx="5"/><text class="num" x="455" y="180" text-anchor="middle">10</text><text class="t-good" x="500" y="180">✓ 문장끼리 맞음</text><path class="line" d="M488 222 H210" marker-end="url(#arC)"/><text class="muted" x="500" y="227">오른쪽 끝부터 맞춘다</text><text class="muted" x="307" y="244" text-anchor="middle">크기 1인 축은 상대 크기만큼 복사된다 (head 8개에 같은 마스크)</text></svg></figure>

### 데모 크기를 `(2, 10, 512)`로 잡은 이유

예시일 뿐이고 문장이 몇 개든 코드는 같다. 축마다 숫자를 다르게 잡아야 `(2, 8, 10, 64)`처럼 모양만 보고 어느 축인지 구분된다. 문장이 1개면 크기 1인 브로드캐스트 축과 헷갈리고, 위의 마스크 실수 같은 batch 관련 버그도 드러나지 않는다.

