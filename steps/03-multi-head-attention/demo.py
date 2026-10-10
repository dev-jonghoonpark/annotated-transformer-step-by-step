import torch
import torch.nn as nn

from transformer import MultiHeadedAttention, subsequent_mask

torch.manual_seed(0)

mha = MultiHeadedAttention(h=8, d_model=512)
x = torch.randn(2, 10, 512)  # 문장 2개, 토큰 10개
out = mha(x, x, x)

print("입력           :", tuple(x.shape))
print("출력           :", tuple(out.shape), " ← 입력과 같은 모양")
print("head별 가중치  :", tuple(mha.attn.shape), " (batch, head, query, key)")
print("head 하나의 d_k:", mha.d_k, "= 512 / 8")
print()

n = sum(p.numel() for p in mha.parameters())
print(f"파라미터 수: {n:,}  = 4 × (512×512 + 512)  (W_Q, W_K, W_V, W_O)")
print()

print("1) 프로젝션 후 head로 쪼개는 모양 변화")
q = mha.linears[0](x)
print("   linear(x)              ", tuple(q.shape))
q = q.view(2, -1, 8, 64)
print("   .view(B, -1, h, d_k)    ", tuple(q.shape))
q = q.transpose(1, 2)
print("   .transpose(1, 2)        ", tuple(q.shape), " ← head가 batch처럼 앞으로")
print()

print("2) PyTorch 내장 nn.MultiheadAttention과 같은 계산인가?")
ours = MultiHeadedAttention(4, 32, dropout=0.0).eval()
ref = nn.MultiheadAttention(32, 4, dropout=0.0, batch_first=True).eval()
with torch.no_grad():
    ref.in_proj_weight.copy_(torch.cat([l.weight for l in ours.linears[:3]]))
    ref.in_proj_bias.copy_(torch.cat([l.bias for l in ours.linears[:3]]))
    ref.out_proj.weight.copy_(ours.linears[3].weight)
    ref.out_proj.bias.copy_(ours.linears[3].bias)
y = torch.randn(1, 6, 32)
mask = subsequent_mask(6)
a = ours(y, y, y, mask=mask)
b, _ = ref(y, y, y, attn_mask=~mask[0])  # PyTorch는 True가 '가림'
print("   최대 차이:", f"{(a - b).abs().max().item():.2e}")
print()

print("3) W_O(linears[-1])는 head들의 결과를 섞는다")
# ours의 forward를 단계별로 다시 따라가며 head별 출력을 꺼낸다 (head 4개, d_k 8)
from transformer import attention

h, d_k = ours.h, ours.d_k
q, k, v = [lin(y).view(1, -1, h, d_k).transpose(1, 2) for lin in ours.linears[:3]]
heads, _ = attention(q, k, v)  # (1, 4, 6, 8)
concat = heads.transpose(1, 2).reshape(1, 6, h * d_k)  # head를 옆으로 이어 붙인 (1, 6, 32)
W_O = ours.linears[-1]
print("   concat 모양:", tuple(concat.shape), " → W_O 통과 →", tuple(W_O(concat).shape))
print("   ours(y, y, y)와 차이:", f"{(W_O(concat) - ours(y, y, y)).abs().max().item():.1e}")
# concat @ W_O 는 'head마다 W_O의 자기 몫(8열)을 곱해 더한 것'과 같다
per_head = [heads[:, i] @ W_O.weight[:, i * d_k:(i + 1) * d_k].T for i in range(h)]
print("   Σ head_i @ W_O의 i번째 블록 + bias 와 차이:",
      f"{(sum(per_head) + W_O.bias - W_O(concat)).abs().max().item():.1e}")
print("   → 출력 한 칸 한 칸이 head 4개 모두의 결과를 섞은 값이다")
