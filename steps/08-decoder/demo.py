import copy

import torch

from transformer import (
    Decoder, DecoderLayer, MultiHeadedAttention, PositionwiseFeedForward, subsequent_mask,
)

torch.manual_seed(0)

d_model, N = 512, 6
c = copy.deepcopy
attn = MultiHeadedAttention(8, d_model)
ff = PositionwiseFeedForward(d_model, 2048, 0.1)
decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), 0.1), N).eval()

print(f"디코더 층 {len(decoder.layers)}개, 층마다 서브층 {len(decoder.layers[0].sublayer)}개 (self-attn, src-attn, FFN)")
print(f"디코더 파라미터 수: {sum(p.numel() for p in decoder.parameters()):,}  ← 어텐션이 하나 더 있어 인코더보다 크다")
print()

memory = torch.randn(1, 7, d_model)  # 인코더 출력 (원문 토큰 7개)
src_mask = torch.ones(1, 1, 7, dtype=torch.bool)
y = torch.randn(1, 5, d_model)  # 지금까지 만든 번역 토큰 5개
out = decoder(y, memory, src_mask, subsequent_mask(5))
print("디코더 입력", tuple(y.shape), "+ memory", tuple(memory.shape), "→ 출력", tuple(out.shape))
print()

print("뒤쪽 토큰(4, 5번째)만 바꾸면 앞쪽 출력은?")
y2 = y.clone()
y2[:, 3:] = torch.randn(1, 2, d_model)
out2 = decoder(y2, memory, src_mask, subsequent_mask(5))
diff = (out - out2).abs().amax(dim=(0, 2))
for i, d in enumerate(diff.tolist()):
    print(f"  {i + 1}번째 토큰 출력 차이: {d:.1e}")
print("  → 미래를 가리는 마스크 덕분에 앞 3개는 전혀 바뀌지 않는다")
print()

print("마스크 없이 돌리면?")
out3 = decoder(y, memory, src_mask, None)
out4 = decoder(y2, memory, src_mask, None)
print(f"  1번째 토큰 출력 차이: {(out3 - out4)[:, 0].abs().max().item():.1e}  ← 미래를 엿봤다")
