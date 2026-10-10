import math

import altair as alt
import pandas as pd
import torch

from show import show
from transformer import Embeddings, PositionalEncoding

torch.manual_seed(0)

emb = Embeddings(d_model=512, vocab=1000)
tokens = torch.tensor([[17, 42, 7]])
e = emb(tokens)
print("토큰 id", tokens.tolist(), "→ 임베딩", tuple(e.shape))
print(f"nn.Embedding 값의 표준편차   : {emb.lut(tokens).std():.3f}")
print(f"× √512 = × {math.sqrt(512):.1f} 한 뒤 : {e.std():.3f}")
print()

pe = PositionalEncoding(d_model=512, dropout=0.0)
print("PE 버퍼:", tuple(pe.pe.shape), " (1, max_len, d_model)")
print("학습되는 파라미터 수:", sum(p.numel() for p in pe.parameters()), " ← register_buffer라 학습하지 않는다")
print()
y = pe(torch.zeros(1, 4, 512))
print("위치 0~3의 앞 6개 차원 (0차원=sin, 1차원=cos, ...)")
print(y[0, :, :6].round(decimals=3))
print()
print("같은 단어라도 위치가 다르면 다른 벡터가 된다")
same = emb(torch.tensor([[42, 42]]))
z = pe(same)
print(f"  임베딩만: 두 벡터 차이 {(same[0, 0] - same[0, 1]).norm():.2f}")
print(f"  PE 더한 뒤: 두 벡터 차이 {(z[0, 0] - z[0, 1]).norm():.2f}")


# 원본 노트북의 example_positional
def example_positional():
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )


show(example_positional(), "positional")
