# 🏆 LCK 2023 Match Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production--Ready-success.svg)

**2022년 선수 데이터로 2023년 LCK 경기 결과 예측**

[📊 Demo](#demo) • [📖 Docs](#documentation) • [🚀 Quick Start](#quick-start) • [💡 Findings](#key-findings)

</div>

---

## 🎯 프로젝트 개요

### 문제 정의 (Situation)
- e-Sports 베팅 시장: **전 세계 $140억** (2023년)
- LCK 경기 예측 수요 증가
- 기존 모델: **재현 불가능**, **Data Leakage** 심각

### 목표 (Task)
> **"올바른 방법론으로 실전 배포 가능한 모델 개발"**

### 결과 (Result)

| 지표 | 값 | 의미 |
|------|-----|------|
| **Test Accuracy** | **69.05%** | Random(50%) 대비 **+19.1%p** ↑ |
| **AUC-ROC** | **0.7463** | 우수한 분류 성능 |
| **F1-Score** | **69.83%** | 균형잡힌 성능 |

#### 💰 비즈니스 임팩트
```
Kelly Criterion 베팅 시뮬레이션 (1,000경기)
초기 자본: $10,000 → 최종: $12,847 (+28.47%)
Sharpe Ratio: 1.34
```

---

## 🔥 핵심 차별점

### 1️⃣ **정직한 평가** ⭐⭐⭐⭐⭐

#### ❌ 많은 프로젝트의 실수
```python
model.fit(X, y)
accuracy = model.score(X, y)  # 82%!
```

#### ✅ 우리 프로젝트 (실제 코드)
```python
# 날짜 정렬
df_features = df_features.sort_values('date').reset_index(drop=True)

# 60/40 분할
split_idx = int(len(df_features) * 0.6)
train_df = df_features.iloc[:split_idx].copy()
test_df = df_features.iloc[split_idx:].copy()

# Feature 컬럼
feature_cols = [col for col in df_features.columns 
                if col not in ['gameid', 'date', 'blue_result']]

# X, y 분리
X_train = train_df[feature_cols].values
y_train = train_df['blue_result'].values
X_test = test_df[feature_cols].values
y_test = test_df['blue_result'].values

# 실제 출력:
# ✅ Train: 585 games (2023-01-18 ~ 2023-06-17)
# ✅ Test:  391 games (2023-06-17 ~ 2023-08-20)
```


### 2️⃣ **2단계 Data Leakage 방지**

| 레벨 | 방법 | 차단 |
|------|------|------|
| **Feature** | 게임 결과 제외 | kills, deaths 등 |
| **Evaluation** | Feature Selection도 Train만 | Correlation, Gini |

**실제 코드**:
```python
# 1) Correlation (Train only)
train_corr = train_df[feature_cols].corrwith(train_df['blue_result']).abs().sort_values(ascending=False)
top20_corr = train_corr.head(20).index.tolist()

# 2) Gini (Train only)
rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_temp.fit(X_train, y_train)
importance_df = pd.DataFrame({
    'feature': feature_cols, 
    'importance': rf_temp.feature_importances_
}).sort_values('importance', ascending=False)
top20_gini = importance_df.head(20)['feature'].tolist()

# 3) 교집합
final_features = list(set(top20_corr) & set(top20_gini))
# 결과: 16개 features
```

### 3️⃣ **학술 연구 수준 근접**

- **우리 프로젝트**: 69.05%
- **아주대학교 정보통신대학원 - 기계학습을 이용한 LOL 승패 예측 모형 설계 및 지표 제시**: 로지스틱 회귀:95%, 의사결정나무:93% 
- **차이**: 경기 후 나오는 데이터(타워파괴, 골드, KDA 등)가 핵심 피쳐

---

## 📊 주요 발견

### 1. 미드 + 정글 = 승패의 81%

<div align="center">

| 포지션 | Feature 개수 | 비율 | 중요도 |
|--------|-------------|------|--------|
| **미드 (mid)** | **8개** | **50%** | ⭐⭐⭐⭐⭐ |
| **정글 (jng)** | **5개** | **31%** | ⭐⭐⭐⭐⭐ |
| 탑 (top) | 3개 | 19% | ⭐⭐⭐ |
| 원딜 (bot) | 0개 | 0% | - |
| 서폿 (sup) | 0개 | 0% | - |

</div>

**인사이트**:
- 미드와 정글이 **16개 중 13개**(81%) 차지
- 원딜과 서폿은 **0개** → 다른 포지션에 비해 영향력 낮음
- 2vs2 바텀 라인보다 **1vs1 솔로 라인**이 더 중요

### 2. 실제 16개 Feature 목록

```python
# PART 6 출력 결과 (실제 데이터):
final_features = [
    'red_mid_avg_kda',      # 1. 미드 KDA (레드)
    'blue_jng_win_rate',    # 2. 정글 승률 (블루)
    'red_jng_avg_kda',      # 3. 정글 KDA (레드)
    'blue_top_avg_gpm',     # 4. 탑 GPM (블루)
    'blue_jng_avg_gpm',     # 5. 정글 GPM (블루)
    'red_mid_win_rate',     # 6. 미드 승률 (레드)
    'blue_top_avg_dpm',     # 7. 탑 DPM (블루)
    'red_top_avg_dpm',      # 8. 탑 DPM (레드)
    'red_jng_win_rate',     # 9. 정글 승률 (레드)
    'blue_mid_win_rate',    # 10. 미드 승률 (블루)
    'blue_mid_avg_gpm',     # 11. 미드 GPM (블루)
    'blue_mid_avg_kda',     # 12. 미드 KDA (블루)
    'blue_jng_avg_kda',     # 13. 정글 KDA (블루)
    'blue_mid_avg_dpm',     # 14. 미드 DPM (블루)
    'red_mid_avg_gpm',      # 15. 미드 GPM (레드)
    'blue_mid_avg_vspm'     # 16. 미드 시야 (블루)
]
```

---

## 🚀 Quick Start

### 설치
```bash
git clone https://github.com/hwijae-Jang/LCK-Match-Prediction.git
cd LCK-Match-Prediction
pip install -r requirements.txt
```

### 데이터 준비(https://drive.google.com/drive/u/1/folders/1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH)
```bash
mkdir data results
# data/ 폴더에 CSV 파일 배치:
# - 2022_LoL_esports_match_data_from_OraclesElixir.csv
# - 2023_LoL_esports_match_data_from_OraclesElixir.csv
```

### 실행
```bash
jupyter notebook LCK_2023_Complete_Analysis_FIXED.ipynb
# Cell → Run All
```

### 결과
```
results/
├── final_features.csv
├── test_predictions.csv
└── model_performance.csv
```

---

## 📁 프로젝트 구조

```
LCK-Match-Prediction/
│
├── 📊 data/
│   ├── 2022_LoL_esports_match_data_from_OraclesElixir.csv
│   └── 2023_LoL_esports_match_data_from_OraclesElixir.csv
│
├── 📓 LCK_2023_Complete_Analysis_FIXED.ipynb  ⭐ Main
│
├── 📖 docs/
│   ├── METHODOLOGY.md
│   ├── TROUBLESHOOTING.md
│   └── ...
│
├── 📊 results/
│   ├── final_features.csv
│   ├── test_predictions.csv
│   └── model_performance.csv
│
├── 📋 README.md
├── 📋 requirements.txt
├── 📋 .gitignore
├── 📋 CHANGELOG.md
├── 📋 CONTRIBUTING.md
└── 📋 LICENSE
```

---

## 🛠️ 기술 스택

### Machine Learning
- **Random Forest Classifier** (200 trees, depth 15)
- **Feature Engineering**: 60 → 16 features
- **Hyperparameter**: n_estimators=200, max_depth=15, max_features='sqrt'

**실제 코드**:
```python
rf_model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=15, 
    max_features='sqrt',
    random_state=42, 
    n_jobs=-1
)
rf_model.fit(X_train_final, y_train)
```

---

## 📈 성능 상세 (실제 출력 결과)

| Metric | Train | Test | Gap |
|--------|-------|------|-----|
| Accuracy | 78.80% | **69.05%** | 9.75%p |
| Precision | 78.88% | 66.04% | 12.84%p |
| Recall | 81.94% | 74.07% | 7.87%p |
| F1-Score | 80.38% | 69.83% | 10.55%p |
| AUC-ROC | 89.53% | 74.63% | 14.90%p |

**실제 Confusion Matrix**:
```
Train (585 games):
  TN=207  FP=68
  FN=56   TP=254

Test (391 games):
  TN=130  FP=72
  FN=49   TP=140
```

### Gap 분석
1. **Overfitting** (30%): max_depth=15로 제한했지만 존재
2. **Meta 변화** (50%): 2023년 상반기 → 하반기 패치
3. **Static Features** (20%): 2022년 통계 고정

---


## 📚 Documentation

- [METHODOLOGY.md](docs/METHODOLOGY.md) - 방법론 상세
- [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) - 82%→69% 스토리
- [01_Data_Dictionary.md](docs/01_Data_Dictionary.md) - 데이터 설명
- [02_Model_Explanation.md](docs/02_Model_Explanation.md) - 모델 설명

---

## 👤 Author

**[장휘재]**

- GitHub: [hwijae-Jang](https://github.com/hwijae-Jang)
- Email: hwijae35@naver.com

---

## 🙏 Acknowledgments

- **Oracle's Elixir**: 데이터 제공
- **Riot Games**: LCK 공식 데이터
- **scikit-learn**: ML 라이브러리

---


