import torch

from transformer import greedy_decode, make_model

torch.manual_seed(0)


# 원본 노트북의 inference_test: 학습하지 않은 모델로 한 토큰씩 생성
def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)
    ys = greedy_decode(test_model, src, src_mask, max_len=10, start_symbol=0)
    print("Example Untrained Model Prediction:", ys)


for _ in range(10):
    inference_test()
print()
print("→ 아직 학습하지 않아서 입력과 상관없는 토큰을 반복한다. 14단계에서 이 모델에 복사를 가르친다")
