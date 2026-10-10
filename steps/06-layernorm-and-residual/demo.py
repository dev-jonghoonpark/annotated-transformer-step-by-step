import torch

from transformer import LayerNorm, SublayerConnection, PositionwiseFeedForward

torch.manual_seed(0)

x = torch.randn(2, 3, 8) * 5 + 3
ln = LayerNorm(8)
y = ln(x)
print("LayerNorm은 토큰 하나(마지막 차원)마다 평균 0, 표준편차 1로 맞춘다")
print("  전 평균  :", [f"{v:.2f}" for v in x.mean(-1)[0].tolist()])
print("  후 평균  :", [f"{abs(v):.2f}" for v in y.mean(-1)[0].tolist()])
print("  후 표준편차:", [f"{v:.2f}" for v in y.std(-1)[0].tolist()])
print("  학습 파라미터:", [n for n, _ in ln.named_parameters()], "(스케일 γ, 이동 β)")
print()

sub = SublayerConnection(8, dropout=0.0)
print("SublayerConnection: x + Sublayer(LayerNorm(x))")
out = sub(x, lambda t: torch.zeros_like(t))
print("  서브층이 0만 내놓으면 출력 = 입력?", torch.allclose(out, x))
print("  → 서브층이 아무것도 못 배워도 신호는 그대로 흐른다 (잔차 연결)")
print()

print("층을 깊게 쌓았을 때 신호 크기 (FFN 20개)")
torch.manual_seed(0)
ffns = [PositionwiseFeedForward(64, 256, dropout=0.0) for _ in range(20)]
subs = [SublayerConnection(64, dropout=0.0) for _ in range(20)]
h_plain = h_res = torch.randn(1, 5, 64)
for f, s in zip(ffns, subs):
    h_plain = f(h_plain)
    h_res = s(h_res, f)
print(f"  그냥 쌓기    : 표준편차 {h_plain.std():.2e}")
print(f"  잔차 + 정규화: 표준편차 {h_res.std():.2e}")
