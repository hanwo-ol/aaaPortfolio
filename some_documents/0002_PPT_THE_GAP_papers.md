## 1) “기존 학습 이론은 ‘더 많이 학습할수록 정교해진다’는 상황에 주로 초점”

### (A) i.i.d. / 고정 분포(정적 환경) 가정이 표준이라는 근거

* **Vapnik (1998)** *Statistical Learning Theory* — ERM/일반화 이론의 기본 틀이 i.i.d. 샘플 및 고정 분포에서의 일반화로 정리됨. ([Springer][1])
* **Shalev-Shwartz & Ben-David (2014)** *Understanding Machine Learning: From Theory to Algorithms* — 일반화/학습 이론의 표준 가정(훈련·테스트 분포 고정, i.i.d.)을 교과서적으로 정리. ([Journal of Machine Learning Research][2])

> 이 축의 핵심: “학습을 더 한다”는 것이 보통 **같은 분포에서 더 잘 맞춘다(수렴/일반화 향상)**는 전제에서 설계돼 있다는 점.

---

## 2) “금융처럼 신호가 약하고 노이즈가 큰 환경은 충분히 설명되지 않음”

### (B) 금융 시계열의 stylized facts: heavy-tail / 비정상성 / 구조 변화

* **Cont (2001)** “Empirical properties of asset returns: stylized facts and statistical issues” — 수익률의 heavy tails, 변동성 클러스터링 등 “표준 i.i.d. 가정과 어긋나는” 대표 성질을 정리. ([Rama][3])
* **(Survey) Major Issues in High-Frequency Financial Data Analysis (2025)** — 고빈도 금융데이터의 핵심 이슈로 **nonstationarity, low signal-to-noise ratio** 등을 명시적으로 열거. ([MDPI][4])

### (C) “금융 시계열은 low SNR”을 직접적으로 말하는 문헌 예시

* **Wang & Ventre (2024)** “A Financial Time Series Denoiser Based on Diffusion Model” — 금융 시계열이 **low signal-to-noise ratio**라 예측/해석이 어렵다고 명시. ([arXiv][5])

> 이 축의 핵심: 금융은 “학습 데이터를 더 파면 진짜 구조가 더 선명해지는 문제”라기보다, **노이즈/구조변화가 기본값인 문제**라는 점을 문헌들이 반복적으로 확인.

---

## 3) “특히 ‘왜 적응을 짧게 해야 하는지’에 대한 이론적 기준이 없음”

(= 메타러닝 inner loop를 ‘몇 step’ 돌릴지에 대한 principled rule이 부족)

“부재”를 정면증명하긴 어려움, 다만 아래 문헌들이 **현실적 근거**를 제공하고 있어서 정리해보면,
(1) inner-loop step/step-size가 민감한 하이퍼파라미터라는 점, (2) multi-step에서 안정성/수렴 조건이 까다롭다는 점, (3) 그래서 다양한 work들이 “안정화/튜닝”을 다루지만 **‘금융 low-SNR에서의 적정 J’ 같은 이론적 기준**은 직접 주지 않는다는 점.

### (D) multi-step MAML은 step 수/step-size에 민감하고 조건이 필요하다는 이론

* **“Theoretical Convergence of Multi-Step Model-Agnostic Meta-Learning” (JMLR)** — N-step MAML의 수렴 보장을 위해 inner-step size가 N(steps)에 반비례해야 한다는 식으로, **multi-step이 단순히 ‘많이 돌리면 좋다’가 아님**을 이론적으로 보여줌. ([jmlr.csail.mit.edu][6])

### (E) 실증적으로도 inner loop hyperparameter sensitivity가 크다는 근거

* **Antoniou et al. (MAML++) / “How to Train Your MAML”** — MAML이 “무엇이 성패를 가르는지”를 다루며 **inner-loop 하이퍼파라미터 민감성/안정화**를 문제로 놓고 개선안을 제시. ([arXiv][7])

### (F) “inner loop이 불필요/민감” 논쟁을 촉발한 이론·분석 계열

* **“Meta-learning with negative learning rates” (2021)** — inner-loop 학습률을 0(=inner loop 없음)까지 포함해 분석하며, **inner loop 자체/강도가 핵심 변수가 될 수 있음**을 보여줌. ([arXiv][8])

> 이 축의 핵심: 메타러닝 커뮤니티에서도 “inner loop를 깊게”가 자동으로 좋은 게 아니고, **step/step-size가 민감하고 안정성 문제가 있다**는 건 문헌으로 이미 확인되어옴.
> 다만 “금융 low-SNR에서 왜/얼마나 짧아야 하는지”를 **명시적 기준으로 주는 이론**은 상대적으로 빈약하다고 생각함.

---

## “짧게 멈추는 게 정당”을 일반 ML 관점에서 받쳐주는 레퍼런스

(이건 메타러닝이 아니라 “반복 최적화를 길게 하면 일반화가 나빠질 수 있고, 멈추는 규칙이 이론적으로 존재한다”는 기반)

* **Yao, Rosasco, Caponnetto (2007)** “On Early Stopping in Gradient Descent Learning” — 반복 최적화에서 iteration 수(멈추는 시점)가 일반화에 영향을 주며, early stopping을 이론적으로 다룸. ([Springer][1])
* **Raskutti, Wainwright, Yu (2014, JMLR)** “Early stopping and non-parametric regression…” — early stopping을 **regularization으로 분석**하고 data-dependent stopping rule을 제시. ([Journal of Machine Learning Research][2])



[1]: https://link.springer.com/article/10.1007/s00365-006-0663-2?utm_source=chatgpt.com "On Early Stopping in Gradient Descent Learning | Constructive Approximation | Springer ..."
[2]: https://jmlr.org/papers/volume15/raskutti14a/raskutti14a.pdf?utm_source=chatgpt.com "Early Stopping and Non-parametric Regression: An Optimal Data-dependent Stopping Rule"
[3]: https://rama.cont.perso.math.cnrs.fr/pdf/empirical.pdf?utm_source=chatgpt.com "Empirical properties of asset returns: stylized facts and statistical issues"
[4]: https://www.mdpi.com/2227-7390/13/3/347?utm_source=chatgpt.com "Major Issues in High-Frequency Financial Data Analysis: A Survey of Solutions - MDPI"
[5]: https://arxiv.org/html/2409.02138v1?utm_source=chatgpt.com "A Financial Time Series Denoiser Based on Diffusion Model"
[6]: https://jmlr.csail.mit.edu/papers/volume23/20-720/20-720.pdf?utm_source=chatgpt.com "Theoretical Convergence of Multi-Step Model-Agnostic Meta-Learning"
[7]: https://arxiv.org/pdf/1810.09502?utm_source=chatgpt.com "HOW TO TRAIN YOUR MAML - arXiv.org"
[8]: https://arxiv.org/abs/2102.00940?utm_source=chatgpt.com "[2102.00940] Meta-learning with negative learning rates - arXiv.org"
