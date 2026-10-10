import torch

from transformer import make_model, subsequent_mask

torch.manual_seed(0)

model = make_model(src_vocab=8316, tgt_vocab=6384)  # 15단계에서 만들 독일어·영어 사전 크기
print("EncoderDecoder의 다섯 부품")
for name, m in model.named_children():
    print(f"  {name:10s} {sum(p.numel() for p in m.parameters()):>12,}  {m.__class__.__name__}")
print(f"  {'합계':10s} {sum(p.numel() for p in model.parameters()):>12,}")
print()

model.eval()
src = torch.randint(4, 8316, (1, 9))
tgt = torch.randint(4, 6384, (1, 5))
src_mask = torch.ones(1, 1, 9, dtype=torch.bool)
memory = model.encode(src, src_mask)
out = model.decode(memory, src_mask, tgt, subsequent_mask(5))
logp = model.generator(out)
print("src", tuple(src.shape), "→ encode →", tuple(memory.shape))
print("tgt", tuple(tgt.shape), "→ decode →", tuple(out.shape))
print("generator →", tuple(logp.shape), " (다음 토큰의 로그 확률)")
print("확률 합:", [f"{v:.3f}" for v in logp.exp().sum(-1)[0].tolist()])
print()
w = model.encoder.layers[0].feed_forward.w_1.weight
print(f"xavier_uniform_ 초기화: w_1 값 범위 ±{w.abs().max():.4f}  (이론 한계 √(6/(512+2048)) = {(6 / (512 + 2048)) ** 0.5:.4f})")
