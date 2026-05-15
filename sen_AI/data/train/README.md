# Train 데이터 폴더

이 폴더에는 모델 학습에 사용하는 train split 파일이 저장됩니다.

필요한 파일은 다음 2개입니다.

```text
X_train.npy
y_train.npy
```

파일 형식:

```text
X_train: (샘플 수, 30, 120)
y_train: (샘플 수,)
```

`X_train.npy`는 전처리된 keypoint 입력 데이터이고, `y_train.npy`는 각 샘플의 정답 label index입니다.

이 파일들은 프로젝트 최상위 폴더에서 다음 명령을 실행하면 생성됩니다.

```bash
python data/prepare_split.py
```

LSTM, GRU, CNN 학습 코드는 모두 이 폴더의 train 데이터를 사용합니다.
