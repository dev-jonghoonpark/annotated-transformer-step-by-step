# 토큰마다 혼자 생각하기: Feed-Forward

- 원본: [위치별 피드포워드 네트워크](https://dev-jonghoonpark.github.io/annotated-transformer-modern/#position-wise-feed-forward-networks)

어텐션이 토큰끼리 정보를 **주고받는** 단계라면, 피드포워드(FFN)는 받은 정보를 토큰마다 **혼자 소화하는** 단계다.

$$\mathrm{FFN}(x)=\max(0, xW_1 + b_1) W_2 + b_2$$

512차원을 2048차원으로 넓혔다가(ReLU) 다시 512로 줄인다. 모든 위치에 같은 가중치를 쓰지만, 위치끼리는 섞지 않는다. 그래서 이름이 "position-wise"다.

코드는 `nn.Linear` 두 개와 ReLU, dropout이 전부다.

## 실행 결과에서 볼 것

- 문장 전체를 넣었을 때 4번째 토큰의 출력과 그 토큰 하나만 넣었을 때의 출력이 같다. 토큰 사이에 정보가 오가지 않는다는 뜻이다.
- 파라미터가 약 210만 개로, 어텐션 블록(105만 개)의 두 배다. Transformer 파라미터의 상당 부분이 FFN에 있다.
