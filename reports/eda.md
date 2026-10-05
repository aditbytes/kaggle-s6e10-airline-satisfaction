# EDA: Predicting Airline Satisfaction

## Shape and target

- Train: **699,635** rows · Test: **299,844** rows · Features: **21**
- Satisfied: **44.36%** (balanced enough; stratify folds anyway)
- Exact duplicate feature rows in train: **0**

## Missing values

| column | train | test |
|---|---|---|
| Arrival Delay in Minutes | 292 | 130 |

LightGBM handles NaN natively; no imputation needed for tree models.

## Single-feature AUC

Each feature on its own, as a ranking score for `satisfaction`.

| feature | auc | direction |
|---|---|---|
| Online boarding | 0.8404 | + |
| Class | 0.7822 | + |
| Inflight entertainment | 0.7445 | + |
| Seat comfort | 0.7321 | + |
| Type of Travel | 0.7027 | + |
| On-board service | 0.6993 | + |
| Flight Distance | 0.6956 | + |
| Cleanliness | 0.6894 | + |
| Leg room service | 0.6877 | + |
| Inflight wifi service | 0.6627 | + |
| Baggage handling | 0.6549 | + |
| Checkin service | 0.6355 | + |
| Age | 0.6244 | + |
| Food and drink | 0.6211 | + |
| Ease of Online booking | 0.6090 | + |
| Customer Type | 0.5862 | + |
| Departure/Arrival time convenient | 0.5265 | - |
| Arrival Delay in Minutes | 0.5162 | - |
| Gate location | 0.5137 | + |
| Departure Delay in Minutes | 0.5085 | - |
| Gender | 0.5056 | + |

## Gender

| Gender | rows | satisfied_rate |
|---|---|---|
| Female | 347,952 | 0.4380 |
| Male | 351,683 | 0.4491 |

## Customer Type

| Customer Type | rows | satisfied_rate |
|---|---|---|
| Loyal Customer | 576,990 | 0.4951 |
| disloyal Customer | 122,645 | 0.2009 |

## Type of Travel

| Type of Travel | rows | satisfied_rate |
|---|---|---|
| Business travel | 497,441 | 0.5843 |
| Personal Travel | 202,194 | 0.0973 |

## Class

| Class | rows | satisfied_rate |
|---|---|---|
| Business | 342,212 | 0.7253 |
| Eco | 327,404 | 0.1676 |
| Eco Plus | 30,019 | 0.2422 |

## Service ratings (0-5) by outcome

| rating | mean if not satisfied | mean if satisfied | gap |
|---|---|---|---|
| Online boarding | 2.72 | 4.17 | 1.45 |
| Inflight entertainment | 2.95 | 4.09 | 1.13 |
| Seat comfort | 3.11 | 4.14 | 1.03 |
| Cleanliness | 2.99 | 3.87 | 0.87 |
| On-board service | 3.13 | 4.00 | 0.86 |
| Leg room service | 3.10 | 3.94 | 0.85 |
| Inflight wifi service | 2.45 | 3.20 | 0.75 |
| Checkin service | 3.15 | 3.76 | 0.61 |
| Food and drink | 3.01 | 3.58 | 0.57 |
| Baggage handling | 3.56 | 4.12 | 0.56 |
| Ease of Online booking | 2.59 | 3.10 | 0.51 |
| Gate location | 2.99 | 3.05 | 0.06 |
| Departure/Arrival time convenient | 3.22 | 3.07 | -0.15 |

## Train vs test drift

Large differences would mean CV may not reflect the leaderboard.

| feature | train mean | test mean | diff % |
|---|---|---|---|
| Age | 39.002 | 38.993 | -0.023 |
| Flight Distance | 1353.779 | 1353.369 | -0.030 |
| Departure Delay in Minutes | 1.176 | 1.157 | -1.670 |
| Arrival Delay in Minutes | 1.134 | 1.120 | -1.245 |
| Inflight wifi service | 2.779 | 2.777 | -0.077 |
| Departure/Arrival time convenient | 3.156 | 3.152 | -0.146 |
| Ease of Online booking | 2.813 | 2.813 | -0.024 |
| Gate location | 3.014 | 3.013 | -0.020 |
| Food and drink | 3.266 | 3.270 | 0.134 |
| Online boarding | 3.364 | 3.366 | 0.046 |
| Seat comfort | 3.569 | 3.572 | 0.089 |
| Inflight entertainment | 3.454 | 3.456 | 0.061 |
| On-board service | 3.517 | 3.513 | -0.116 |
| Leg room service | 3.473 | 3.469 | -0.123 |
| Baggage handling | 3.805 | 3.802 | -0.080 |
| Checkin service | 3.420 | 3.422 | 0.040 |
| Cleanliness | 3.380 | 3.385 | 0.140 |
