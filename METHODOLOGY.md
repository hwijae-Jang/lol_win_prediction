# 📖 방법론 (Methodology)

## 1. 데이터 수집

- **제공**: Oracle's Elixir
- **2022년**: 50,127 rows × 164 columns
- **2023년**: 55,843 rows × 164 columns
- **필터링**: LCK 리그만

```python
lck_2022 = df_2022[df_2022['league'] == 'LCK'].copy()
lck_2023 = df_2023[df_2023['league'] == 'LCK'].copy()
```

---

## 2. Feature Engineering

### 2.1 Raw Features (60개)

**6가지 지표 × 2팀 × 5포지션 = 60**

```python
# 6가지 지표
1. win_rate   = wins / games
2. avg_kda    = (K + A) / (D + 1)
3. avg_gpm    = earnedgold / (gamelength / 60)
4. avg_dpm    = damagetochampions / (gamelength / 60)
5. avg_vspm   = visionscore / (gamelength / 60)
6. avg_kp     = (K + A) / team_kills
```

### 2.2 Feature Selection (16개)

**실제 코드 (PART 6)**:
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
# 결과: 16개
```

**실제 16개 Feature**:
```
1.  red_mid_avg_kda      (미드 KDA)
2.  blue_jng_win_rate    (정글 승률)
3.  red_jng_avg_kda      (정글 KDA)
4.  blue_top_avg_gpm     (탑 GPM)
5.  blue_jng_avg_gpm     (정글 GPM)
6.  red_mid_win_rate     (미드 승률)
7.  blue_top_avg_dpm     (탑 DPM)
8.  red_top_avg_dpm      (탑 DPM)
9.  red_jng_win_rate     (정글 승률)
10. blue_mid_win_rate    (미드 승률)
11. blue_mid_avg_gpm     (미드 GPM)
12. blue_mid_avg_kda     (미드 KDA)
13. blue_jng_avg_kda     (정글 KDA)
14. blue_mid_avg_dpm     (미드 DPM)
15. red_mid_avg_gpm      (미드 GPM)
16. blue_mid_avg_vspm    (미드 시야)
```

**포지션 분포 (실제 출력)**:
```
TOP: 3개 (19%)
JNG: 5개 (31%)
MID: 8개 (50%)  ← 가장 중요!
BOT: 0개
SUP: 0개
```

---

## 3. Train/Test Split ⭐⭐⭐

**Time-based Split (60/40) - 실제 코드 (PART 5)**:

```python
print("\n[PART 5] ⭐ TIME-BASED TRAIN/TEST SPLIT")
print("-" * 80)

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
```

**실제 출력**:
```
✅ Train: 585 games (2023-01-18 08:17:31 ~ 2023-06-17 08:56:41)
✅ Test:  391 games (2023-06-17 08:56:41 ~ 2023-08-20 08:17:24)

   X_train: (585, 60)
   X_test:  (391, 60)
```

---

## 4. 모델 학습

**실제 코드 (PART 7)**:
```python
print("\n[PART 7] 모델 학습 및 평가")
print("-" * 80)

# 최종 Feature로 필터링
X_train_final = train_df[final_features].values
X_test_final = test_df[final_features].values

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=15, 
    max_features='sqrt',
    random_state=42, 
    n_jobs=-1
)
rf_model.fit(X_train_final, y_train)

# 예측
y_train_pred = rf_model.predict(X_train_final)
y_train_proba = rf_model.predict_proba(X_train_final)[:, 1]
y_test_pred = rf_model.predict(X_test_final)
y_test_proba = rf_model.predict_proba(X_test_final)[:, 1]

print("✅ 학습 및 예측 완료!")
```

---

## 5. 평가

**실제 코드 (PART 7)**:
```python
def evaluate_model(y_true, y_pred, y_proba, dataset_name):
    print(f"\n{'='*60}")
    print(f"{dataset_name} Performance")
    print(f"{'='*60}")
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    logloss = log_loss(y_true, y_proba)
    
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1-Score:   {f1:.4f}")
    print(f"AUC-ROC:    {auc:.4f}")
    print(f"Log Loss:   {logloss:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
    print(f"  FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")
    
    return {...}

# 평가
train_metrics = evaluate_model(y_train, y_train_pred, y_train_proba, "TRAIN SET")
test_metrics = evaluate_model(y_test, y_test_pred, y_test_proba, "TEST SET (진짜!)")
```

---

## 최종 결과 (실제 출력)

```
============================================================
TRAIN SET Performance
============================================================
Accuracy:   0.7880
Precision:  0.7888
Recall:     0.8194
F1-Score:   0.8038
AUC-ROC:    0.8953
Log Loss:   0.3746

Confusion Matrix:
  TN=207  FP= 68
  FN= 56  TP=254

============================================================
TEST SET (진짜!) Performance
============================================================
Accuracy:   0.6905
Precision:  0.6604
Recall:     0.7407
F1-Score:   0.6983
AUC-ROC:    0.7463
Log Loss:   1.4568

Confusion Matrix:
  TN=130  FP= 72
  FN= 49  TP=140
```

---

**방법론의 투명성 = 프로젝트의 신뢰성**
