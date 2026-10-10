# CUDA GPU가 있어야 학습할 수 있다. 웹 페이지의 출력은 RTX 3070에서 실행한 기록이다.
import torch

from transformer import (
    check_outputs, create_dataloaders, load_tokenizers, load_trained_model, load_vocab, make_model,
)

spacy_de, spacy_en = load_tokenizers()
vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)

# multi30k_model_final.pt가 없으면 8 epoch 학습한다 (RTX 3070에서 약 13분)
load_trained_model(vocab_src, vocab_tgt, spacy_de, spacy_en)


# 원본 노트북의 run_model_example: 학습한 가중치를 CPU로 불러 검증 문장을 번역한다
def run_model_example(n_examples=5):
    print("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=1,
        is_distributed=False,
    )

    print("Loading Trained Model ...")

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(
        torch.load("multi30k_model_final.pt", map_location=torch.device("cpu"))
    )

    print("Checking Model Outputs:")
    example_data = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples
    )
    return model, example_data


run_model_example(n_examples=1)
