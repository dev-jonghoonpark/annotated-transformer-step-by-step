# 진짜 데이터로 배치 만들기

- 원본: [이터레이터](https://dev-jonghoonpark.github.io/annotated-transformer-modern/#iterators)

토큰화한 문장을 텐서로 바꿔 배치로 묶는다.

- **`collate_batch`**: 문장마다 `<s>` + 단어 id들 + `</s>`를 이어 붙이고, `max_padding` 길이까지 `<blank>`(2)로 채운다. 그다음 배치로 쌓는다(`torch.stack`). DataLoader가 배치를 만들 때마다 이 함수를 부른다.
- **`create_dataloaders`**: train/valid용 DataLoader를 만든다. 여러 GPU로 나눠 학습할 때는 `DistributedSampler`를 쓴다.

이 배치를 11단계의 `Batch`로 감싸면 학습에 필요한 `tgt`, `tgt_y`, 마스크가 생긴다. 14단계 복사 과제에서 `data_gen`이 하던 일을 진짜 데이터로 하는 것이다.

## 실행 결과에서 볼 것

- 문장 길이는 제각각이지만 모두 16칸으로 맞춰져 있고, 뒤는 2(`<blank>`)로 채워져 있다.
- id를 사전으로 되돌리면 원래 문장이 나온다.
- 세 번째 문장은 16칸보다 길어서 **뒤가 잘렸다**. `</s>`도 없다. `max_padding - len(...)`이 음수가 되면 `pad`가 텐서를 잘라 버리기 때문이다. 원본 코드의 `# warning - overwrites values for negative values of padding - len` 주석이 이 현상을 가리킨다. 실제 학습에서는 `max_padding=72`라 거의 일어나지 않는다.
