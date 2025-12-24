# e-sports LoL 승패 예측 (선수 경기 스탯 기반)



## 데이터
- `data/Player_data/*.csv`: 선수별 경기 스탯 테이블
- 라벨 생성 규칙(원본 노트북 그대로):
  - `Win/Loss = 1 if P value != 0 else 0`

## 빠른 실행 (재현용)
```bash
pip install -r requirements.txt
python scripts/train_eval.py --csv data/Player_data/T1_Faker.csv --seed 42 --cv 3
```

실행 후 `reports/`에 생성:
- `report.md` (정확도, confusion matrix, best params 등)
- `feature_importance.png` (변수 중요도 Top 15)
- `corr_heatmap.png` (상관관계 히트맵)

## 여러 선수 파일 한번에 요약 (선택)
```bash
python scripts/batch_eval.py --dir data/Player_data --seed 42 --cv 3 --out reports/batch_summary.csv
```

## 원본 노트북
원본 폴더의 핵심 노트북 일부는 `notebooks_archive/`에 보관했습니다(참고용).
