import torch

from transformer import Batch, create_dataloaders, load_tokenizers, load_vocab

torch.manual_seed(0)

spacy_de, spacy_en = load_tokenizers()
vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
print()

train_dl, valid_dl = create_dataloaders(
    torch.device("cpu"), vocab_src, vocab_tgt, spacy_de, spacy_en,
    batch_size=3, max_padding=16, is_distributed=False,
)
src, tgt = next(iter(train_dl))
print("한 배치 (batch_size=3, max_padding=16)")
print("src", tuple(src.shape))
print(src)
print("tgt", tuple(tgt.shape))
print(tgt)
print()
itos_src, itos_tgt = vocab_src.get_itos(), vocab_tgt.get_itos()
print("id를 다시 단어로")
for i in range(3):
    print("  de:", " ".join(itos_src[t] for t in src[i]))
    print("  en:", " ".join(itos_tgt[t] for t in tgt[i]))
print()
b = Batch(src, tgt, pad=2)
print("Batch로 감싸면 학습에 필요한 마스크가 붙는다")
print("  src_mask", tuple(b.src_mask.shape), " tgt_mask", tuple(b.tgt_mask.shape), " ntokens", b.ntokens.item())
print()
print(f"학습 배치 수: {len(train_dl):,} (batch_size=3일 때). 실제 학습은 batch_size=32, max_padding=72")
