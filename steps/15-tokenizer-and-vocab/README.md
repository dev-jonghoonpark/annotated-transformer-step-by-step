# 진짜 데이터: 토큰화와 사전

- 원본: [3부: 실전 예제](https://dev-jonghoonpark.github.io/annotated-transformer-modern/#part-3-a-real-world-example) · [데이터 불러오기](https://dev-jonghoonpark.github.io/annotated-transformer-modern/#data-loading)

3부에서는 Multi30k 독일어→영어 번역을 학습한다. 이미지 설명 문장 약 3만 쌍이라 논문의 WMT(450만 쌍)보다 훨씬 작지만, 전체 흐름을 보기에는 충분하다.

- **`load_tokenizers`**: spaCy의 독일어·영어 모델로 문장을 단어 단위로 자른다.
- **`build_vocabulary`**: train+valid+test 전체에서 두 번 이상 나온 단어로 사전을 만든다. 맨 앞 네 칸은 특수 토큰이다: `<s>`(0, 시작), `</s>`(1, 끝), `<blank>`(2, 패딩), `<unk>`(3, 모르는 단어). 그래서 앞 단계 코드들이 `pad=2`, `start_symbol=0`을 기본값으로 쓴다.
- **`load_vocab`**: 만든 사전을 `vocab.pt`에 저장해 두고 다음부터는 불러온다.

> 원본은 `torchtext`의 `Multi30k`, `build_vocab_from_iterator`, `to_map_style_dataset`을 쓴다. torchtext는 개발이 중단돼 최신 torch에서 import조차 되지 않는다. 그래서 같은 이름·같은 동작의 대체품을 파일 안에 넣었다(diff의 `# torchtext 대신` 부분). 데이터는 같은 분할을 Hugging Face Hub의 `bentrevett/multi30k`에서 받는다. torch 2.6부터 `torch.load`의 기본값이 바뀌어서 사전을 불러올 때 `weights_only=False`가 필요하다.

## 실행 결과에서 볼 것

- 29,000 / 1,014 / 1,000 문장쌍.
- 사전 크기: 독일어 8,316, 영어 6,384. 원본이 torchtext로 만든 것과 같은 순서다.
- 사전은 특수 토큰 4개 다음에 가장 흔한 토큰부터 온다. 독일어는 `.`, `Ein`, `einem`, 영어는 `a`, `.`, `A` 순서다. 대소문자를 구분하므로 `a`와 `A`가 다른 단어다.
