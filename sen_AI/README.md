# 수어 문징 인식 모델 학습 및 평가

이 프로젝트는 전처리된 수어 keypoint 데이터 `(30, 120)`을 입력으로 사용해 단어 하나를 예측하는 모델을 학습하고 평가합니다.

현재 포함된 모델은 다음 3개입니다.

- LSTM
- GRU
- 1D-CNN

각 모델은 같은 train/validation 데이터를 사용하며, 클래스 수는 `data/validation/classes.npy`의 길이를 기준으로 자동 설정됩니다.

## 폴더 구조

```text
.
├─ README.md
├─ requirements.txt
├─ compare_metrics.py
├─ data/
│  ├─ prepare_split.py
│  ├─ raw/
│  ├─ train/
│  ├─ validation/
│  └─ test/
├─ lstm/
│  ├─ lstm.py
│  └─ report_metrics_lstm.py
├─ gru/
│  ├─ gru.py
│  └─ report_metrics_gru.py
├─ cnn/
│  ├─ cnn.py
│  └─ report_metrics_cnn.py
└─ results/
   ├─ lstm/
   ├─ gru/
   └─ cnn/
```

## 설치

필요한 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

`requirements.txt`에는 다음 라이브러리가 포함되어 있습니다.

```text
numpy
pandas
scikit-learn
tensorflow
matplotlib
seaborn
```

## 데이터 준비

`data/raw/`에 원본 batch 파일과 `classes.npy`를 넣습니다.

필요한 파일 형식은 다음과 같습니다.

```text
Batch_001_X.npy
Batch_001_y.npy
...
Batch_033_X.npy
Batch_033_y.npy
classes.npy
```

`data/prepare_split.py`는 `used_videos_final_CLEANED.csv`의 `video_uid`에서 `SENxxxx` 그룹을 추출하고, `StratifiedGroupKFold`로 train/validation을 분리합니다.

같은 `SENxxxx` 그룹은 train과 validation에 동시에 들어가지 않습니다.

split 생성:

```bash
python data/prepare_split.py
```

생성되는 파일:

```text
data/train/X_train.npy
data/train/y_train.npy
data/validation/X_validation.npy
data/validation/y_validation.npy
data/validation/classes.npy
```

입력 shape는 다음 형식을 사용합니다.

```text
X: (샘플 수, 30, 120)
y: (샘플 수,)
```

## 모델 학습

각 모델을 학습합니다.

```bash
python lstm/lstm.py
python gru/gru.py
python cnn/cnn.py
```

학습 후 저장되는 파일:

```text
lstm/lstm_best.keras
gru/gru_best.keras
cnn/cnn_best.keras
```

학습 로그는 `results` 폴더에 저장됩니다.

```text
results/lstm/history.csv
results/gru/history.csv
results/cnn/history.csv
```

## 모델 평가

학습된 모델을 validation 데이터로 평가합니다.

```bash
python lstm/report_metrics_lstm.py
python gru/report_metrics_gru.py
python cnn/report_metrics_cnn.py
```

각 모델의 평가 결과는 `results/모델명/`에 저장됩니다.

```text
results/lstm/metrics.csv
results/lstm/classification_report.csv
results/lstm/confusion_top.csv
results/lstm/sample_predictions.csv
results/lstm/learning_curve.png
results/lstm/confusion_matrix.png
results/lstm/top_confusions.png
```

GRU와 CNN도 같은 구조로 저장됩니다.

평가 파일이 생성하는 항목은 다음과 같습니다.

- `metrics.csv`: 전체 loss, accuracy, top3 accuracy, macro F1, weighted F1, latency
- `classification_report.csv`: 클래스별 precision, recall, f1-score, support
- `confusion_top.csv`: 많이 헷갈린 정답/예측 단어 쌍
- `sample_predictions.csv`: validation 샘플 일부의 정답, 예측, confidence, top3 결과
- `learning_curve.png`: 학습 과정의 loss/accuracy 그래프
- `confusion_matrix.png`: validation confusion matrix
- `top_confusions.png`: 많이 틀린 단어 쌍 상위 그래프


## 모델 비교

각 모델의 `metrics.csv`를 하나로 합칩니다.

```bash
python compare_metrics.py
```

생성되는 파일:

```text
results/model_comparison.csv
```


## 실행 순서 요약

```bash
python data/prepare_split.py

python lstm/lstm.py
python gru/gru.py
python cnn/cnn.py

python lstm/report_metrics_lstm.py
python gru/report_metrics_gru.py
python cnn/report_metrics_cnn.py

python compare_metrics.py
```
