import torch

from transformer import data_gen, example_simple_model

torch.manual_seed(0)

print("복사 과제의 데이터 한 배치 (앞 3문장)")
b = next(data_gen(V=11, batch_size=3, nbatches=1))
print("  src  ", b.src.tolist())
print("  tgt_y", b.tgt_y.tolist(), " ← src를 그대로 따라 쓰는 게 정답")
print()

print("원본의 example_simple_model 실행: N=2 모델, 20 epoch (CPU 약 2분)")
print("한 epoch에 한 줄씩, 그 epoch 두 번째 배치의 손실이 찍힌다")
print()
example_simple_model()
print()
print("입력 [0, 1, ..., 9]를 그대로 따라 썼다면 학습 성공")
print()
print("─" * 60)
print("왜 [0, 1, ..., 9]를 완벽하게 따라 쓰지 못할까?")
print("학습 데이터는 항상 1로 시작하고 1~10만 쓴다. 0은 이 과제에서 패딩 id라 한 번도 입력으로 나온 적이 없다.")
print("같은 설정으로 다시 학습해서 학습 때와 같은 모양의 입력을 넣어 본다.")
print()

import contextlib
import io

from torch.optim.lr_scheduler import LambdaLR

from transformer import (
    DummyOptimizer, DummyScheduler, LabelSmoothing, SimpleLossCompute,
    greedy_decode, make_model, rate, run_epoch,
)

V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9)
lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda step: rate(step, 512, 1.0, 400))
for epoch in range(20):
    model.train()
    with contextlib.redirect_stdout(io.StringIO()):
        run_epoch(data_gen(V, 80, 20), model, SimpleLossCompute(model.generator, criterion),
                  optimizer, lr_scheduler, mode="train")
model.eval()


def copy(seq):
    src = torch.LongTensor([seq])
    out = greedy_decode(model, src, torch.ones(1, 1, len(seq)), max_len=len(seq), start_symbol=seq[0])
    return out[0].tolist()


print("  원본 테스트  [0..9]  →", copy(list(range(10))))
print("  학습과 같은 [1..10] →", copy(list(range(1, 11))))
ok = 0
for _ in range(100):
    seq = torch.randint(1, V, (10,)).tolist()
    seq[0] = 1
    ok += copy(seq) == seq
print(f"  학습과 같은 분포의 무작위 문장 100개 중 완벽하게 따라 쓴 것: {ok}개")
