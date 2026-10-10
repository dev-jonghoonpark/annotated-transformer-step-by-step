import torch

from transformer import Batch

# 0 = <s>, 1 = </s>, 2 = <blank>(패딩)
src = torch.tensor([[0, 11, 12, 13, 1, 2]])
tgt = torch.tensor([[0, 21, 22, 1, 2, 2]])
b = Batch(src, tgt, pad=2)

print("src      ", b.src.tolist())
print("src_mask ", b.src_mask.int().tolist(), " (패딩만 가림)")
print()
print("tgt      ", tgt.tolist())
print("b.tgt    ", b.tgt.tolist(), " ← 디코더 입력: 마지막 토큰을 뺀 것")
print("b.tgt_y  ", b.tgt_y.tolist(), " ← 정답: 첫 토큰을 뺀 것 (한 칸 앞으로 민 것)")
print()
print("i번째 입력을 보고 i번째 정답을 맞힌다:")
for i, (a, y) in enumerate(zip(b.tgt[0].tolist(), b.tgt_y[0].tolist())):
    print(f"  위치 {i}: {b.tgt[0, : i + 1].tolist()} 까지 보고 → {y}")
print()
print("tgt_mask = 미래 가리기 AND 패딩 가리기")
print(b.tgt_mask[0].int())
print("ntokens (패딩이 아닌 정답 수):", b.ntokens.item())
