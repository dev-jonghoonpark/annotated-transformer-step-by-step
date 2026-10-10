import altair as alt
import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR

from show import show
from transformer import rate

print("rate(step, model_size=512, factor=1, warmup=4000)")
for step in (1, 1000, 2000, 4000, 8000, 20000):
    print(f"  step {step:6d}: {rate(step, 512, 1, 4000):.2e}")
print("→ 4000스텝까지 직선으로 오르고, 그 뒤로 1/√step으로 내려간다")
print()


# 원본 노트북의 example_learning_schedule
def example_learning_schedule():
    opts = [
        [512, 1, 4000],  # example 1
        [512, 1, 8000],  # example 2
        [256, 1, 4000],  # example 3
    ]

    dummy_model = torch.nn.Linear(1, 1)
    learning_rates = []

    # we have 3 examples in opts list.
    for idx, example in enumerate(opts):
        # run 20000 epoch for each example
        optimizer = torch.optim.Adam(
            dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step: rate(step, *example)
        )
        tmp = []
        # take 20K dummy training steps, save the learning rate at each step
        for step in range(20000):
            tmp.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
        learning_rates.append(tmp)

    learning_rates = torch.tensor(learning_rates)

    # Enable altair to handle more than 5000 rows
    alt.data_transformers.disable_max_rows()

    opts_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Learning Rate": learning_rates[warmup_idx, :],
                    "model_size:warmup": ["512:4000", "512:8000", "256:4000"][
                        warmup_idx
                    ],
                    "step": range(20000),
                }
            )
            for warmup_idx in [0, 1, 2]
        ]
    )

    return (
        alt.Chart(opts_data[opts_data.step % 20 == 0])
        .mark_line()
        .properties(width=600)
        .encode(x="step", y="Learning Rate", color="model_size\\:warmup:N")
        .interactive()
    )


show(example_learning_schedule(), "learning_rate")
