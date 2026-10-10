# 17단계에서 학습한 multi30k_model_final.pt가 있어야 한다. 웹 페이지의 출력은 RTX 3070에서 실행한 기록이다.
import altair as alt
import torch

from show import show
from transformer import (
    check_outputs, create_dataloaders, get_decoder_self, get_decoder_src, get_encoder,
    load_tokenizers, load_vocab, make_model, visualize_layer,
)

spacy_de, spacy_en = load_tokenizers()
vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)


def run_model_example(n_examples=5):
    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"), vocab_src, vocab_tgt, spacy_de, spacy_en,
        batch_size=1, is_distributed=False,
    )
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(
        torch.load("multi30k_model_final.pt", map_location=torch.device("cpu"))
    )
    example_data = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples
    )
    return model, example_data


# 원본 노트북의 viz_encoder_self / viz_decoder_self / viz_decoder_src
def viz_encoder_self():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[
        len(example_data) - 1
    ]  # batch object for the final example

    layer_viz = [
        visualize_layer(
            model, layer, get_encoder, len(example[1]), example[1], example[1]
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        # & layer_viz[1]
        & layer_viz[2]
        # & layer_viz[3]
        & layer_viz[4]
        # & layer_viz[5]
    )


def viz_decoder_self():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[len(example_data) - 1]

    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_self,
            len(example[1]),
            example[1],
            example[1],
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        & layer_viz[1]
        & layer_viz[2]
        & layer_viz[3]
        & layer_viz[4]
        & layer_viz[5]
    )


def viz_decoder_src():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[len(example_data) - 1]

    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_src,
            max(len(example[1]), len(example[2])),
            example[1],
            example[2],
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        & layer_viz[1]
        & layer_viz[2]
        & layer_viz[3]
        & layer_viz[4]
        & layer_viz[5]
    )


show(viz_encoder_self(), "encoder_self")
show(viz_decoder_self(), "decoder_self")
show(viz_decoder_src(), "decoder_src")
