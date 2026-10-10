from transformer import Multi30k, load_tokenizers, load_vocab, tokenize

spacy_de, spacy_en = load_tokenizers()

train, val, test = Multi30k(language_pair=("de", "en"))
print(f"Multi30k: train {len(train):,} / valid {len(val):,} / test {len(test):,} 문장쌍")
print()
de, en = train[0]
print("첫 문장쌍")
print("  de:", de)
print("  en:", en)
print("  de 토큰:", tokenize(de, spacy_de))
print("  en 토큰:", tokenize(en, spacy_en))
print()

vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
print()
print("독일어 사전 앞 20개:", vocab_src.get_itos()[:20])
print("영어 사전 앞 20개 :", vocab_tgt.get_itos()[:20])
print()
tokens = tokenize(en, spacy_en)
print("영어 문장 → id:", vocab_tgt(tokens))
print("사전에 없는 단어 'Transformerology' →", vocab_tgt["Transformerology"], "(<unk>)")
