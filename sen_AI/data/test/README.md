# Test 데이터 폴더

이 폴더는 선택 사항입니다.

`report_metrics_*.py` 실행 시 이 폴더에 test 파일이 있으면 validation 성능과 함께 test 성능도 계산합니다.

필요한 파일은 다음 2개입니다.

```text
X_test.npy
y_test.npy
```

파일 형식:

```text
X_test: (샘플 수, 30, 120)
y_test: (샘플 수,)
```

test 데이터는 train/validation 데이터와 같은 전처리 방식을 사용해야 합니다.

이 폴더에 `X_test.npy`, `y_test.npy`가 없으면 test 평가는 건너뛰고 validation 평가만 수행합니다.
