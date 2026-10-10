import math

import torch

from transformer import attention

torch.manual_seed(0)

# 토큰 3개짜리 문장 하나. 토큰마다 4차원 벡터.
x = torch.randn(1, 3, 4)

# self-attention: query, key, value를 모두 같은 x로
out, p_attn = attention(x, x, x)

print("입력 x       :", tuple(x.shape), " (batch, 토큰 수, d_k)")
print("출력         :", tuple(out.shape))
print("가중치 p_attn:", tuple(p_attn.shape), " (batch, query 토큰, key 토큰)")
print()
print("어텐션 가중치 — i행은 'i번째 토큰이 각 토큰을 얼마나 참고하는가'")
print(p_attn[0].round(decimals=3))
print("행마다 합:", [f"{v:.3f}" for v in p_attn[0].sum(-1).tolist()])
print()
print("출력 0번 토큰 = 가중치로 섞은 value")
manual = sum(p_attn[0, 0, j] * x[0, j] for j in range(3))
print("  attention() :", [f"{v:.4f}" for v in out[0, 0].tolist()])
print("  직접 계산    :", [f"{v:.4f}" for v in manual.tolist()])
print()

print("왜 √d_k로 나누는가 — 성분이 평균 0·분산 1인 q, k의 내적은 분산이 d_k가 된다")
for d_k in (4, 64, 512):
    q, k = torch.randn(10000, d_k), torch.randn(10000, d_k)
    dots = (q * k).sum(-1)
    print(
        f"  d_k={d_k:3d}  q·k 표준편차 {dots.std():6.2f}"
        f"  →  √d_k로 나누면 {(dots / math.sqrt(d_k)).std():.2f}"
    )
print()
q, k = torch.randn(1, 512), torch.randn(10, 512)
raw = (q @ k.T).softmax(-1)
scaled = (q @ k.T / math.sqrt(512)).softmax(-1)
print("d_k=512, key 10개일 때 softmax 결과")
print("  나누지 않으면:", " ".join(f"{v:.3f}" for v in raw[0].tolist()))
print("  √d_k로 나누면:", " ".join(f"{v:.3f}" for v in scaled[0].tolist()))
print("  → 나누지 않으면 한 칸에 거의 다 몰려서(포화) 기울기가 사라진다")
