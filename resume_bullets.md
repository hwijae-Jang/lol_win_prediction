## 피그마 이력서 업데이트용 문장(복붙)

**e-sports 리그오브레전드 승패 예측 | 팀 프로젝트(3인)**  
- 선수 경기 스탯(예: KDA, 골드, 오브젝트 지표 등) 기반 승패 예측 모델 구축(RandomForest + GridSearchCV)
- (재현 가능) `T1_Faker.csv` 기준 테스트 정확도 **96.7%** 재현 가능(seed=42, report 자동 생성)
- 변수 중요도(Feature importance) 및 상관관계 히트맵 시각화로 주요 영향 변수 분석
- 실행 1줄 재현: `python scripts/train_eval.py --csv data/Player_data/T1_Faker.csv --seed 42 --cv 3`


