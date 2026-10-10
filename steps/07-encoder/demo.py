import copy

import torch

from transformer import Encoder, EncoderLayer, MultiHeadedAttention, PositionwiseFeedForward

torch.manual_seed(0)

d_model, N = 512, 6
c = copy.deepcopy
attn = MultiHeadedAttention(8, d_model)
ff = PositionwiseFeedForward(d_model, 2048, 0.1)
encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), 0.1), N).eval()

print(f"인코더 층 {len(encoder.layers)}개, 층마다 서브층 {len(encoder.layers[0].sublayer)}개 (self-attn, FFN)")
n = sum(p.numel() for p in encoder.parameters())
print(f"인코더 파라미터 수: {n:,}")
print("층마다 별도 가중치?", encoder.layers[0].self_attn.linears[0].weight is not encoder.layers[1].self_attn.linears[0].weight)
print()

x = torch.randn(1, 6, d_model)
mask = torch.tensor([[[True, True, True, True, False, False]]])  # 뒤 2개는 패딩
out = encoder(x, mask)
print("입력", tuple(x.shape), "→ 출력", tuple(out.shape))
print()

print("패딩 자리 값을 마음대로 바꿔도 진짜 토큰의 출력은 그대로인가?")
x2 = x.clone()
x2[:, 4:] = torch.randn(1, 2, d_model) * 100
out2 = encoder(x2, mask)
print(f"  진짜 토큰 4개 출력 차이: {(out[:, :4] - out2[:, :4]).abs().max().item():.1e}")
print("  → 마스크 덕분에 어텐션이 패딩을 보지 않는다")
