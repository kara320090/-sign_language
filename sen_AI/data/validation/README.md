# Validation 데이터 폴더

이 폴더에는 모델 학습 중 검증과 학습 후 성능 평가에 사용하는 validation split 파일이 저장됩니다.

필요한 파일은 다음 3개입니다.

```text
X_validation.npy
y_validation.npy
classes.npy
```

파일 형식:

```text
X_validation: (샘플 수, 30, 120)
y_validation: (샘플 수,)
classes: (클래스 수,)
```

`classes.npy`는 label index를 실제 단어 텍스트로 바꾸기 위해 사용합니다.

모델 학습 코드와 평가 코드는 `classes.npy`의 길이를 기준으로 클래스 수를 자동 계산합니다.

이 파일들은 프로젝트 최상위 폴더에서 다음 명령을 실행하면 생성됩니다.

```bash
python data/prepare_split.py
```

`report_metrics_*.py` 파일들은 이 폴더의 validation 데이터를 기준으로 성능을 계산합니다.
