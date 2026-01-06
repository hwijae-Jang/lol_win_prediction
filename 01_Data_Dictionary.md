# 📊 Data Dictionary (데이터 설명)

## 목차
1. [데이터 개요](#데이터-개요)
2. [원본 데이터셋](#원본-데이터셋)
3. [Feature 정의](#feature-정의)
4. [최종 Feature 16개](#최종-feature-16개)
5. [데이터 통계](#데이터-통계)

---

## 1. 데이터 개요

### 📥 데이터 출처
- **제공**: Oracle's Elixir (https://oracleselixir.com)(https://drive.google.com/drive/u/1/folders/1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH)
- **데이터**: 공식 Riot API 기반
- **범위**: 전 세계 프로 리그 (LCK, LPL, LCS, LEC 등)

### 📅 데이터 기간
```
2022년: 50,127 rows × 164 columns
2023년: 55,843 rows × 164 columns

필터링 후 (LCK만):
- 2022년: 약 2,000+ rows
- 2023년: 976 games (Train 585 + Test 391)
```

---

## 2. 원본 데이터셋

### 2.1 기본 정보

| 컬럼 | 설명 | 예시 |
|------|------|------|
| `gameid` | 고유 게임 ID | "ESPORTSTMNT06_2451748" |
| `datacompleteness` | 데이터 완전성 | "complete" |
| `league` | 리그 이름 | "LCK" |
| `year` | 연도 | 2023 |
| `split` | 시즌 | "Spring" |
| `date` | 경기 날짜 | "2023-01-18 08:17:31" |
| `game` | 경기 번호 | 1, 2, 3 |
| `patch` | 게임 패치 | "13.1" |

### 2.2 팀 정보

| 컬럼 | 설명 | 예시 |
|------|------|------|
| `side` | 팀 색상 | "Blue" / "Red" |
| `teamname` | 팀 이름 | "T1", "Gen.G" |
| `teamid` | 팀 ID | "oe:team:..." |
| `result` | 승패 | 1 (승리) / 0 (패배) |

### 2.3 선수 정보

| 컬럼 | 설명 | 예시 |
|------|------|------|
| `playername` | 선수 이름 | "Faker", "Chovy" |
| `playerid` | 선수 ID | "oe:player:..." |
| `position` | 포지션 | "mid", "jng", "top" |
| `champion` | 챔피언 | "Azir", "LeBlanc" |

### 2.4 게임 통계

| 카테고리 | 주요 컬럼 | 설명 |
|----------|-----------|------|
| **전투** | `kills`, `deaths`, `assists` | KDA |
| **골드** | `earnedgold`, `totalgold` | 획득 골드 |
| **데미지** | `damagetochampions` | 챔피언 데미지 |
| **시야** | `visionscore`, `wardskilled` | 시야 점수 |
| **오브젝트** | `dragons`, `barons`, `towers` | 용, 바론, 타워 |
| **CS** | `minionkills`, `monsterkills` | 미니언, 정글 몬스터 |

---

## 3. Feature 정의

### 3.1 생성한 Feature (60개)

**공식**: 6가지 지표 × 2팀 × 5포지션 = 60개

#### 📊 6가지 지표

1. **win_rate**: 승률
   ```python
   win_rate = wins / total_games
   # 예: 20승 10패 → 0.667 (66.7%)
   ```

2. **avg_kda**: 평균 KDA
   ```python
   kda = (kills + assists) / (deaths + 1)
   # 예: K=5, A=10, D=2 → (5+10)/(2+1) = 5.0
   ```

3. **avg_gpm**: 분당 골드 획득량 (Gold Per Minute)
   ```python
   gpm = earnedgold / (gamelength / 60)
   # 예: 15,000 골드 / 30분 = 500 GPM
   ```

4. **avg_dpm**: 분당 데미지 (Damage Per Minute)
   ```python
   dpm = damagetochampions / (gamelength / 60)
   # 예: 18,000 데미지 / 30분 = 600 DPM
   ```

5. **avg_vspm**: 분당 시야 점수 (Vision Score Per Minute)
   ```python
   vspm = visionscore / (gamelength / 60)
   # 예: 60 시야점수 / 30분 = 2.0 VSPM
   ```

6. **avg_kp**: 킬 관여율 (Kill Participation)
   ```python
   kp = (kills + assists) / team_total_kills
   # 예: K=3, A=5, 팀킬=15 → 8/15 = 53.3%
   ```

#### 🎯 Feature 명명 규칙

```
{team}_{position}_{metric}

team: blue, red
position: top, jng, mid, bot, sup
metric: win_rate, avg_kda, avg_gpm, avg_dpm, avg_vspm, avg_kp

예시:
- blue_mid_win_rate: 블루팀 미드 선수의 승률
- red_jng_avg_gpm: 레드팀 정글 선수의 평균 GPM
```

---

## 4. 최종 Feature 16개

### 4.1 Feature Selection 과정

```python
# 1단계: Correlation (상위 20개)
train_corr = train_df[feature_cols].corrwith(train_df['blue_result']).abs()
top20_corr = train_corr.sort_values(ascending=False).head(20)

# 2단계: Gini Importance (상위 20개)
rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
rf_temp.fit(X_train, y_train)
top20_gini = importance_df.sort_values('importance', ascending=False).head(20)

# 3단계: 교집합
final_features = list(set(top20_corr) & set(top20_gini))
# 결과: 16개
```

### 4.2 최종 16개 Feature 상세

| # | Feature 이름 | 팀 | 포지션 | 지표 | 설명 |
|---|-------------|-----|--------|------|------|
| 1 | `red_mid_avg_kda` | Red | Mid | KDA | 레드팀 미드의 평균 KDA |
| 2 | `blue_jng_win_rate` | Blue | Jungle | 승률 | 블루팀 정글의 승률 |
| 3 | `red_jng_avg_kda` | Red | Jungle | KDA | 레드팀 정글의 평균 KDA |
| 4 | `blue_top_avg_gpm` | Blue | Top | GPM | 블루팀 탑의 평균 GPM |
| 5 | `blue_jng_avg_gpm` | Blue | Jungle | GPM | 블루팀 정글의 평균 GPM |
| 6 | `red_mid_win_rate` | Red | Mid | 승률 | 레드팀 미드의 승률 |
| 7 | `blue_top_avg_dpm` | Blue | Top | DPM | 블루팀 탑의 평균 DPM |
| 8 | `red_top_avg_dpm` | Red | Top | DPM | 레드팀 탑의 평균 DPM |
| 9 | `red_jng_win_rate` | Red | Jungle | 승률 | 레드팀 정글의 승률 |
| 10 | `blue_mid_win_rate` | Blue | Mid | 승률 | 블루팀 미드의 승률 |
| 11 | `blue_mid_avg_gpm` | Blue | Mid | GPM | 블루팀 미드의 평균 GPM |
| 12 | `blue_mid_avg_kda` | Blue | Mid | KDA | 블루팀 미드의 평균 KDA |
| 13 | `blue_jng_avg_kda` | Blue | Jungle | KDA | 블루팀 정글의 평균 KDA |
| 14 | `blue_mid_avg_dpm` | Blue | Mid | DPM | 블루팀 미드의 평균 DPM |
| 15 | `red_mid_avg_gpm` | Red | Mid | GPM | 레드팀 미드의 평균 GPM |
| 16 | `blue_mid_avg_vspm` | Blue | Mid | VSPM | 블루팀 미드의 평균 VSPM |

### 4.3 포지션별 분포

```
MID (미드): 8개 (50%)
  - blue_mid_win_rate, blue_mid_avg_gpm, blue_mid_avg_kda
  - blue_mid_avg_dpm, blue_mid_avg_vspm
  - red_mid_win_rate, red_mid_avg_kda, red_mid_avg_gpm

JNG (정글): 5개 (31%)
  - blue_jng_win_rate, blue_jng_avg_gpm, blue_jng_avg_kda
  - red_jng_avg_kda, red_jng_win_rate

TOP (탑): 3개 (19%)
  - blue_top_avg_gpm, blue_top_avg_dpm
  - red_top_avg_dpm

BOT (원딜): 0개 (0%)

SUP (서폿): 0개 (0%)
```

---

## 5. 데이터 통계

### 5.1 Train/Test Split

```
전체 데이터: 976 games (LCK 2023)

Train (60%): 585 games
  - 기간: 2023-01-18 ~ 2023-06-17
  - Shape: (585, 60) → (585, 16) after feature selection

Test (40%): 391 games
  - 기간: 2023-06-17 ~ 2023-08-20
  - Shape: (391, 60) → (391, 16) after feature selection
```

### 5.2 타겟 변수 (y)

```
blue_result:
  - 1: 블루팀 승리
  - 0: 레드팀 승리 (= 블루팀 패배)

분포:
  - 전체: 약 50:50 (균형)
  - Train: 310 승 / 275 패 (53% / 47%)
  - Test: 189 승 / 202 패 (48% / 52%)
```

### 5.3 Feature 통계 (예시)

| Feature | Min | Max | Mean | Std |
|---------|-----|-----|------|-----|
| `blue_mid_win_rate` | 0.30 | 0.75 | 0.52 | 0.08 |
| `blue_mid_avg_kda` | 1.5 | 8.0 | 3.8 | 1.2 |
| `blue_mid_avg_gpm` | 350 | 550 | 420 | 45 |
| `blue_mid_avg_dpm` | 400 | 800 | 580 | 90 |

---

## 💡 Q&A

### Q1: Feature가 왜 60개인가요?

**A**: 6가지 지표 × 2팀 × 5포지션 = 60개입니다.

```
6가지: win_rate, avg_kda, avg_gpm, avg_dpm, avg_vspm, avg_kp
2팀: blue, red
5포지션: top, jng, mid, bot, sup
```

### Q2: 왜 16개만 선택했나요?

**A**: Correlation과 Gini Importance의 교집합입니다.

```
Correlation 상위 20개 ∩ Gini 상위 20개 = 16개

이유:
1. Overfitting 방지
2. 해석 가능성 향상
3. 중요한 Feature만 사용
```

### Q3: 원딜과 서폿이 0개인 이유는?

**A**: 두 지표 모두 상위 20개에 들지 못했습니다.

```
가능한 이유:
1. 2vs2 라인 (개인 영향력 분산)
2. 팀 의존성 높음
3. 2022년 통계로는 예측 어려움
4. 시너지 변수 필요
```

---

