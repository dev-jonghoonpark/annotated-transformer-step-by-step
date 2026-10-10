# annotated-transformer-step-by-step

[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)(harvardnlp)의 코드를 **빈 파일에서 시작해 18단계에 걸쳐 한 조각씩 쌓아 가며** 읽는 웹 튜토리얼이다.

**📖 <https://dev-jonghoonpark.github.io/annotated-transformer-step-by-step/>**

원본 노트북은 전체 구조(`EncoderDecoder`)부터 보여 줘서 처음 읽을 때 어디서부터 봐야 할지 막막하다. 이 페이지는 순서를 거꾸로 한다. 어텐션 함수 한 줄에서 시작해 부품을 하나씩 더하고, 단계마다 다음 세 가지를 나란히 보여 준다.

- **바뀐 코드**: 이전 단계 `transformer.py`와의 diff. 탭을 바꾸면 그 시점의 파일 전체를 볼 수 있다.
- **설명**: 무엇을 왜 더하는지, diff에서 어디를 보면 되는지. 원본 [한국어 번역판](https://dev-jonghoonpark.github.io/annotated-transformer-modern/)의 해당 절 링크도 붙였다.
- **실행 결과**: 그 단계 코드로 `demo.py`를 실제로 돌린 출력과 차트. 17·18단계(GPU 학습)만 [annotated-transformer-modern](https://github.com/dev-jonghoonpark/annotated-transformer-modern) 노트북을 RTX 3070에서 실행한 기록을 가져왔다.

📚 시리즈: [how-ai-works](https://github.com/dev-jonghoonpark/how-ai-works)의 Transformer 섹션. 논문 개념은 [How Attention Works](https://github.com/dev-jonghoonpark/how-attention-works), 원본 노트북 전체는 [annotated-transformer-modern](https://github.com/dev-jonghoonpark/annotated-transformer-modern).

## 단계

| | 단계 | 추가되는 코드 |
| --- | --- | --- |
| **1부 · 모델** | 01 어텐션 한 줄부터 | `attention` |
| | 02 보면 안 되는 칸 가리기 | `attention(mask=)`, `subsequent_mask` |
| | 03 Multi-Head Attention | `clones`, `MultiHeadedAttention`, `attention(dropout=)` |
| | 04 Feed-Forward | `PositionwiseFeedForward` |
| | 05 임베딩과 위치 인코딩 | `Embeddings`, `PositionalEncoding` |
| | 06 LayerNorm과 잔차 연결 | `LayerNorm`, `SublayerConnection` |
| | 07 인코더 | `EncoderLayer`, `Encoder` |
| | 08 디코더 | `DecoderLayer`, `Decoder` |
| | 09 전체 모델 | `EncoderDecoder`, `Generator`, `make_model` |
| | 10 greedy decoding | `greedy_decode` |
| **2부 · 학습** | 11 학습용 배치 | `Batch` |
| | 12 label smoothing | `LabelSmoothing`, `SimpleLossCompute` |
| | 13 학습률 스케줄과 학습 루프 | `rate`, `TrainState`, `run_epoch` |
| | 14 복사 과제 | `data_gen`, `example_simple_model` |
| **3부 · 번역** | 15 토큰화와 사전 | spaCy, `build_vocabulary` (torchtext 대체 포함) |
| | 16 배치 만들기 | `collate_batch`, `create_dataloaders` |
| | 17 Multi30k 학습 | `train_worker`, `load_trained_model`, `check_outputs` |
| | 18 어텐션 시각화 | `visualize_layer` 외 |

마지막 단계의 `transformer.py`는 원본 `the_annotated_transformer.py`에서 모델·학습·데이터 코드만 모은 것이다. 함수·클래스 45개 중 41개가 원본과 글자 하나까지 같다. 나머지는 아래와 같고, 해당 단계 설명에도 적었다.

- torchtext → 같은 동작의 대체 코드 `Multi30k`, `Vocab`, `build_vocab_from_iterator`, `to_map_style_dataset` (15단계). torchtext는 최신 torch에서 import되지 않는다. 그래서 `build_vocabulary`, `create_dataloaders`의 `datasets.Multi30k(...)` 호출이 `Multi30k(...)`가 됐다.
- `load_trained_model`은 노트북 전역 변수 대신 사전·토크나이저를 인자로 받는다 (17단계).
- `train_distributed_model`에서 노트북 전용 줄 `from the_annotated_transformer import train_worker`를 뺐다. 노트북에서 `mp.spawn`에 함수를 넘기려고 넣은 줄이다.
- 원본에서 `inference_test` 안에 있던 생성 반복문은 10단계에서 같은 코드인 `greedy_decode`로 먼저 소개한다.

## 구조

```
steps/NN-이름/
  README.md         설명 (첫 줄이 제목)
  transformer.py    이 단계까지의 코드 (tools/make_snapshots.py가 생성)
  demo.py           이 단계에서 실행한 코드
  output.txt        demo.py 출력 ([[chart:이름]] 자리에 차트)
  charts/*.json     Vega-Lite 차트
  RECORDED.md       (17·18단계만) 출력이 실행 기록이라는 표시
tools/
  make_snapshots.py 코드 조각 → 단계별 transformer.py
  run_step.py       demo.py 실행 → output.txt, charts/
  extract_recorded.py  17·18단계 출력을 annotated-transformer-modern 노트북에서 추출
site/               build.py(정적 사이트 생성), style.css, app.js
docs/               빌드 결과 (GitHub Pages)
```

## 다시 만들기

[uv](https://docs.astral.sh/uv/)로 관리한다.

```bash
uv sync
uv run python tools/make_snapshots.py      # 코드 조각을 고쳤다면
uv run python tools/run_step.py all        # 데모 다시 실행 (14단계 약 5분, 15단계는 Multi30k 다운로드)
uv run python site/build.py                # docs/ 생성
python -m http.server -d docs              # 미리 보기
```

단계 하나만 돌려 보려면 `uv run python tools/run_step.py 03`.

## 라이선스

코드는 원본과 같은 MIT 라이선스(`LICENSE`)를 따른다. 원본: [harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer) — Sasha Rush(2018), Austin Huang·Suraj Subramanian·Jonathan Sum·Khalid Almubarak·Stella Biderman(2022 개정).
