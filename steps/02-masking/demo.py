import altair as alt
import pandas as pd
import torch

from show import show
from transformer import attention, subsequent_mask

torch.manual_seed(0)

print("subsequent_mask(5) — True인 칸만 볼 수 있다")
print(subsequent_mask(5)[0].int())
print()

x = torch.randn(1, 5, 8)
_, p = attention(x, x, x, mask=subsequent_mask(5))
print("마스크를 건 self-attention 가중치 — 대각선 위(미래)가 전부 0")
print(p[0].round(decimals=2))
print()

# 패딩 마스크: 문장 길이를 맞추려고 채운 <blank> 자리를 가린다
src = torch.tensor([[5, 7, 9, 2, 2]])  # 2 = <blank>
pad_mask = (src != 2).unsqueeze(-2)
print("패딩 마스크 (src != pad):", pad_mask.int().tolist())
_, p = attention(x, x, x, mask=pad_mask)
print("패딩 자리(마지막 두 열)의 가중치:", p[0, :, 3:].abs().max().item())
print()


# 원본 노트북의 example_mask
def example_mask():
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                    "Window": y,
                    "Masking": x,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect()
        .properties(height=250, width=250)
        .encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )


show(example_mask(), "mask")
