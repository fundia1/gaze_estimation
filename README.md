# **데이터 분석 ➔ [EDA.ipynb](EDA.ipynb)**
# **학습 및 결과 ➔ [main.py](main.py)**

---

## 데이터 폴더 구조


`python -m preprocess.mpii`를 실행하면 `data/` 아래에 다음이 생긴다. `data/`는 `.gitignore`에 들어 있어 저장소에 올라가지 않는다. 변환 과정은 [preprocess/README.md](preprocess/README.md)에 정리되어 있다.

```
data/
├── raw/
│   └── MPIIGaze.tar.gz                  # 원본 아카이브 (2.0G). 있으면 다시 받지 않는다
├── MPIIGaze/                            # 압축 해제한 원본 (5.2G)
│   ├── ReadMe.txt
│   ├── 6 points-based face model.mat
│   ├── Annotation Subset/p00.txt … p14.txt
│   ├── Data/
│   │   ├── Original/p00 … p14/          # 전처리 입력 (4.4G)
│   │   │   ├── Calibration/             # Camera.mat, monitorPose.mat, screenSize.mat
│   │   │   └── day01 … dayNN/           # 0001.jpg … (웹캠 프레임) + annotation.txt
│   │   └── Normalized/p00 … p14/        # day01.mat … (데이터셋 제공 정규화본, 쓰지 않음)
│   └── Evaluation Subset/
│       ├── ReadMe.txt
│       ├── annotation for face image/p00.txt … p14.txt
│       └── sample list for eye image/p00.txt … p14.txt   # 변환할 샘플 목록
└── EyeBased/
    └── MPIIGaze/                        # 전처리 결과 (190M), 학습에 쓰는 데이터
        ├── Image/p00 … p14/1.jpg … 3000.jpg   # 36×60 grayscale 눈 이미지
        └── Label/p00.label … p14.label        # 샘플당 한 줄, 첫 줄은 헤더
```

- 전처리가 읽는 원본은 `Data/Original`의 `Calibration/Camera.mat`(카메라 내부 행렬), 각 day의 `annotation.txt`와 프레임, 그리고 `Evaluation Subset/sample list for eye image`의 목록뿐이다. 나머지는 데이터셋에 함께 들어 있는 파일이다.
- day 폴더 수는 사람마다 다르다 (7~69개).
- 결과는 사람당 3,000개(왼눈 1,500 + 오른눈 1,500), 15명 합계 45,000개다. 오른눈 이미지는 좌우 반전되어 왼눈처럼 저장된다.
- 이미지 파일 이름은 목록 순서대로 붙인 일련번호이며, 원본 프레임 위치는 라벨의 `Origin` 칸에 남는다.

라벨 예 (`Label/p00.label`, 소수는 줄임):

```
Image Origin WhichEye 3DHead 2DGaze 2DHead Rmat Smat GazeOrigin
p00/1.jpg day08/0069.jpg left -0.507,-0.112,-0.029 0.247,-0.052 -0.115,0.508 -0.023,-0.171,-0.037 1.0,1.0,1.205 0.0,0.0,600.0
```

`reader.py`는 `loader("data/EyeBased/MPIIGaze/Label/p00.label", "data/EyeBased/MPIIGaze/Image")`처럼 라벨 파일과 `Image/` 폴더를 받는다. 각 칸의 의미는 [preprocess/README.md의 5. 출력](preprocess/README.md#5-출력)을 본다.


## 과제

눈 이미지와 head pose가 주어졌을 때 시선 방향 `2DGaze`를 추정한다.

### 입력과 출력

| 구분 | 라벨 칸 | 형태 | 내용 |
|------|---------|------|------|
| 입력 | `Image` | (1, 36, 60) float32, 0~1 | 정규화된 grayscale 눈 이미지 |
| 입력 | `2DHead` | (2,) float32 | head pose (yaw, pitch), 라디안 |
| 출력 | `2DGaze` | (2,) float32 | gaze (yaw, pitch), 라디안 |

- 모델은 이미지와 head pose **둘 다** 입력으로 받는다.
- 라벨의 나머지 칸(`Origin`, `WhichEye`, `3DHead`, `Rmat`, `Smat`, `GazeOrigin`)은 입력으로 쓰지 않는다.
- 왼눈과 오른눈은 서로 다른 샘플로 다룬다. 오른눈은 전처리에서 이미 좌우 반전되어 있으므로 두 눈을 같은 모델로 학습한다.

`reader.py`가 이 형태 그대로 돌려준다.

```python
info, label = dataset[i]
info["eye"]        # 입력 이미지
info["head_pose"]  # 입력 head pose
label              # 정답 2DGaze
```

### 평가: leave-one-person-out

15명(p00~p14)을 한 명씩 돌아가며 테스트 대상으로 삼는 15-fold 교차 검증이다.

1. fold k에서는 `pk`의 3,000개를 테스트 세트로, 나머지 14명의 42,000개를 학습 세트로 쓴다.
2. 모델을 학습 세트로 처음부터 학습하고, 테스트 세트에서 평균 각도 오차를 잰다.
3. 15개 fold의 결과를 모두 보고하고, 그 평균을 최종 성능으로 삼는다.

```python
persons = [f"p{k:02d}" for k in range(15)]
label = lambda p: f"data/EyeBased/MPIIGaze/Label/{p}.label"

for test in persons:
    train = reader.txtload([label(p) for p in persons if p != test], "data/EyeBased/MPIIGaze/Image", batch_size=..., shuffle=True)
    test_loader = reader.txtload(label(test), "data/EyeBased/MPIIGaze/Image", batch_size=..., shuffle=False)
    ...
```

주의할 점:

- 테스트하는 사람의 데이터는 학습, 정규화 통계 계산, early stopping, 하이퍼파라미터 선택 어디에도 쓰지 않는다. 검증 세트가 필요하면 학습용 14명 안에서 사람 단위로 떼어 낸다.
- 하이퍼파라미터는 모든 fold에서 같게 둔다. fold마다 따로 고르지 않는다.
- fold마다 모델을 새로 초기화한다.

### 평가 지표: 각도 오차

예측과 정답 (yaw, pitch)를 각각 3D 단위 벡터로 바꾼 뒤 두 벡터 사이의 각도를 도(°) 단위로 잰다. yaw와 pitch의 오차를 따로 재거나 2D 거리로 재지 않는다.

```python
import numpy as np
from preprocess.mpii_norm import gaze_to_3d

def angular_error(pred, gt):
    """pred, gt: (N, 2) (yaw, pitch) 라디안 → (N,) 도"""
    a, b = gaze_to_3d(pred), gaze_to_3d(gt)
    cos = np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
```

결과는 다음처럼 보고한다.

| 테스트 대상 | p00 | p01 | … | p14 | 평균 |
|-------------|-----|-----|---|-----|------|
| 각도 오차 (°) | | | | | |

### Baseline: 선형 회귀

[baseline_linear.py](baseline_linear.py)는 numpy만으로 만든 ridge 회귀다. 눈 이미지 픽셀 2,160개와 `2DHead` 2개를 표준화해 이어 붙이고, 정규 방정식을 풀어 `2DGaze`를 바로 예측한다. 평균·표준편차와 회귀 계수는 fold마다 학습용 14명으로만 계산한다. 전체 15 fold가 CPU에서 20초 안에 끝난다.

```bash
python baseline_linear.py                # alpha=1000, 이미지 + 2DHead
python baseline_linear.py --no-head      # 이미지만
```

leave-one-person-out 각도 오차 (°):

| 방법 | p00 | p01 | p02 | p03 | p04 | p05 | p06 | p07 | p08 | p09 | p10 | p11 | p12 | p13 | p14 | 평균 |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|
| 평균 gaze 예측 | 8.43 | 8.63 | 9.00 | 9.60 | 8.97 | 9.73 | 10.34 | 9.49 | 9.61 | 8.97 | 8.66 | 8.43 | 8.60 | 9.00 | 10.75 | 9.21 |
| ridge 회귀 (alpha=1000) | 5.31 | 7.76 | 6.55 | 8.05 | 6.40 | 7.35 | 7.59 | 7.97 | 7.89 | 8.51 | 6.93 | 6.63 | 7.07 | 7.85 | 7.20 | **7.27** |

- 평균 gaze 예측은 입력을 보지 않고 학습용 14명의 `2DGaze` 평균을 항상 내놓는 하한선이다.
- `2DHead`를 빼면(`--no-head`) 평균 7.28°로, 선형 모델에서는 head pose가 거의 도움이 되지 않는다.
- alpha에 따른 평균 오차: 10 → 7.42°, 100 → 7.37°, 1000 → 7.27°, 10000 → 7.16°, 100000 → 7.05°. 이 값은 테스트 결과로 비교한 것이므로 여기서 alpha를 고르면 테스트 데이터를 보고 하이퍼파라미터를 정하는 셈이 된다. 기본값은 1000으로 둔다.


## Installation (conda)

```
conda create -n gaze python=3.12 -y
conda activate gaze

conda install -c conda-forge jupyter matplotlib # for visualization and Eda
```
