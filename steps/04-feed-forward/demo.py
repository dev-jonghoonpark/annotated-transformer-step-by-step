import torch

from transformer import PositionwiseFeedForward

torch.manual_seed(0)

ffn = PositionwiseFeedForward(d_model=512, d_ff=2048, dropout=0.0)
x = torch.randn(2, 10, 512)
print("입력:", tuple(x.shape), "→ w_1:", tuple(ffn.w_1(x).shape), "→ 출력:", tuple(ffn(x).shape))
n = sum(p.numel() for p in ffn.parameters())
print(f"파라미터 수: {n:,}  (512×2048+2048) + (2048×512+512)")
print()

print("'위치별(position-wise)' = 토큰마다 따로, 같은 가중치로 계산한다")
alone = ffn(x[:, 3:4])  # 4번째 토큰만 넣어도
print("  전체 문장에서 4번째 토큰 출력과 혼자 넣었을 때 출력의 차이:",
      f"{(ffn(x)[:, 3:4] - alone).abs().max().item():.1e}")
print("  → 토큰끼리 정보를 섞는 건 어텐션뿐이고, FFN은 각 토큰을 혼자 변환한다")
print()

h = ffn.w_1(x).relu()
print(f"ReLU 뒤 0이 된 비율: {(h == 0).float().mean().item():.0%}")
