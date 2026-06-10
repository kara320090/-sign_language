import cv2
import numpy as np
from functools import lru_cache
import os
import subprocess
import tempfile
from pathlib import Path

_LAST_EXTRACTOR_STATUS = {
    "method": "unknown",
    "error": ""
}


def _landmarks_to_xyc(landmarks, target_count: int):
    """
    MediaPipe landmark를 [x, y, confidence] 형태로 변환한다.
    없으면 0으로 채우고, 많으면 앞 target_count개만 사용한다.
    """
    result = []

    if landmarks is None:
        return [0.0] * (target_count * 3)

    landmark_list = list(landmarks.landmark)

    for lm in landmark_list[:target_count]:
        visibility = getattr(lm, "visibility", 1.0)
        result.extend([float(lm.x), float(lm.y), float(visibility)])

    current_count = len(result) // 3

    if current_count < target_count:
        result.extend([0.0] * ((target_count - current_count) * 3))

    return result


def _make_411d_from_mediapipe_results(results):
    """
    공통 411D keypoint 구성.

    현재 구성:
    pose 25점 * 3 = 75
    left hand 21점 * 3 = 63
    right hand 21점 * 3 = 63
    face 70점 * 3 = 210

    total = 75 + 63 + 63 + 210 = 411

    이 순서를 기준으로:
    - word_AI는 75:201 구간을 사용하여 30F x 126D hands 입력 생성
    - sentence_AI는 hands + pose 일부를 사용하여 30F x 120D 입력 생성
    - degree_AI는 201:411 face 구간을 사용하여 1F x 280D 입력 생성
    """
    pose = _landmarks_to_xyc(results.pose_landmarks, 25)
    left_hand = _landmarks_to_xyc(results.left_hand_landmarks, 21)
    right_hand = _landmarks_to_xyc(results.right_hand_landmarks, 21)
    face = _landmarks_to_xyc(results.face_landmarks, 70)

    full_411d = pose + left_hand + right_hand + face

    if len(full_411d) != 411:
        raise ValueError(f"411D 생성 실패: 현재 길이 {len(full_411d)}")

    return full_411d


def _load_mediapipe_holistic():
    """
    mediapipe 설치 상태가 환경마다 달라서 안전하게 로딩한다.
    실패하면 None 반환.
    """
    try:
        import mediapipe as mp

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic"):
            return mp.solutions.holistic

        return None

    except Exception:
        return None


def _extract_with_mediapipe(video_path: str, target_frames: int = 30):
    """
    MediaPipe Holistic으로 실제 pose/hand/face keypoint를 추출한다.
    """
    mp_holistic = _load_mediapipe_holistic()

    if mp_holistic is None:
        raise RuntimeError("현재 mediapipe에서 solutions.holistic을 사용할 수 없습니다.")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("영상 프레임 수를 확인할 수 없습니다.")

    frame_indices = np.linspace(0, total_frames - 1, target_frames).astype(int)
    frame_index_set = set(frame_indices.tolist())

    sequence = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    ) as holistic:
        current_idx = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if current_idx in frame_index_set:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                full_411d = _make_411d_from_mediapipe_results(results)
                sequence.append(full_411d)

            current_idx += 1

    cap.release()

    if len(sequence) == 0:
        raise ValueError("영상에서 MediaPipe keypoint를 추출하지 못했습니다.")

    while len(sequence) < target_frames:
        sequence.append(sequence[-1])

    return sequence[:target_frames]


def _extract_video_dependent_fallback(video_path: str, target_frames: int = 30):
    """
    MediaPipe가 현재 환경에서 작동하지 않을 때 사용하는 안전 fallback.

    실제 landmark는 아니지만, 업로드된 영상 프레임의 밝기/대비/움직임 정보를 이용해
    30F x 411D 형태를 생성한다.

    목적:
    - 서버가 죽지 않게 함
    - 프론트-백엔드-영상처리 흐름 유지
    - 모델 입력 shape 유지
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("영상 프레임 수를 확인할 수 없습니다.")

    frame_indices = np.linspace(0, total_frames - 1, target_frames).astype(int)
    frame_index_set = set(frame_indices.tolist())

    sequence = []
    prev_small = None
    current_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if current_idx in frame_index_set:
            resized = cv2.resize(frame, (64, 64))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))

            if prev_small is None:
                motion = 0.0
            else:
                motion = float(np.mean(np.abs(gray - prev_small)))

            prev_small = gray

            base = np.zeros(411, dtype=np.float32)

            # pose 영역 0:75
            base[0:75] = brightness

            # left hand 영역 75:138
            base[75:138] = contrast

            # right hand 영역 138:201
            base[138:201] = motion

            # face 영역 201:411
            sampled = cv2.resize(gray, (21, 10)).flatten()  # 210개
            base[201:411] = sampled[:210]

            sequence.append(base.tolist())

        current_idx += 1

    cap.release()

    if len(sequence) == 0:
        raise ValueError("영상에서 fallback sequence를 생성하지 못했습니다.")

    while len(sequence) < target_frames:
        sequence.append(sequence[-1])

    return sequence[:target_frames]


_LAST_EXTRACTOR_STATUS = {
    "method": "unknown",
    "error": ""
}


@lru_cache(maxsize=16)
def extract_411d_sequence_from_video(video_path: str, target_frames: int = 30):
    """
    mp4/avi/mov/mkv/webm 영상에서 30F x 411D sequence를 추출한다.

    1순위: 현재 backend .venv의 MediaPipe
    2순위: 별도 .venv_mp의 MediaPipe worker
    3순위: OpenCV fallback
    """
    global _LAST_EXTRACTOR_STATUS

    try:
        sequence = _extract_with_mediapipe(video_path, target_frames)
        _LAST_EXTRACTOR_STATUS = {
            "method": "mediapipe_in_backend_venv",
            "error": ""
        }
        return sequence

    except Exception as first_error:
        try:
            sequence = _extract_with_external_mediapipe_worker(video_path, target_frames)
            _LAST_EXTRACTOR_STATUS = {
                "method": "mediapipe_external_worker",
                "error": ""
            }
            return sequence

        except Exception as second_error:
            sequence = _extract_video_dependent_fallback(video_path, target_frames)
            _LAST_EXTRACTOR_STATUS = {
                "method": "opencv_fallback",
                "error": (
                    f"backend mediapipe error: {first_error}; "
                    f"external worker error: {second_error}"
                )
            }
            return sequence


def get_last_extractor_status():
    return _LAST_EXTRACTOR_STATUS


def summarize_keypoint_sequence(sequence):
    """
    추출된 411D sequence의 상태 요약.
    Swagger/프론트 디버깅용.
    """
    if not sequence:
        return {
    "sequence_length": int(arr.shape[0]),
    "frame_dim": int(arr.shape[1]),
    "has_pose": bool(np.any(pose)),
    "has_left_hand": bool(np.any(left)),
    "has_right_hand": bool(np.any(right)),
    "has_face": bool(np.any(face)),
    "extractor_status": get_last_extractor_status(),
}

    arr = np.asarray(sequence, dtype=np.float32)

    pose = arr[:, 0:75]
    left = arr[:, 75:138]
    right = arr[:, 138:201]
    face = arr[:, 201:411]

    return {
        "sequence_length": int(arr.shape[0]),
        "frame_dim": int(arr.shape[1]),
        "has_pose": bool(np.any(pose)),
        "has_left_hand": bool(np.any(left)),
        "has_right_hand": bool(np.any(right)),
        "has_face": bool(np.any(face)),
        "extractor_status": get_last_extractor_status(),
    }


def extract_hands_126_from_411d(sequence_411d):
    """
    word_AI 보고서 기준 입력:
    30F x 126D hands keypoint

    현재 411D 구성:
    pose 0:75
    left hand 75:138
    right hand 138:201
    face 201:411

    hands 126D = left hand 63D + right hand 63D
    """
    arr = np.asarray(sequence_411d, dtype=np.float32)

    if arr.ndim != 2 or arr.shape[1] != 411:
        raise ValueError(f"411D sequence shape 오류: {arr.shape}")

    hands_126 = arr[:, 75:201].astype(np.float32)

    if hands_126.shape != (30, 126):
        raise ValueError(f"word_AI 입력 shape 오류: {hands_126.shape}, expected (30, 126)")

    return hands_126


def extract_sentence_120_from_411d(sequence_411d):
    """
    sentence_AI 입력용 30F x 120D sequence 생성.

    구성:
    left hand 21점 x,y = 42
    right hand 21점 x,y = 42
    pose 18점 x,y = 36

    total = 120D
    """
    arr = np.asarray(sequence_411d, dtype=np.float32)

    if arr.ndim != 2 or arr.shape[1] != 411:
        raise ValueError(f"411D sequence shape 오류: {arr.shape}")

    pose_75 = arr[:, 0:75].reshape(arr.shape[0], 25, 3)
    left_63 = arr[:, 75:138].reshape(arr.shape[0], 21, 3)
    right_63 = arr[:, 138:201].reshape(arr.shape[0], 21, 3)

    left_xy = left_63[:, :, :2].reshape(arr.shape[0], 42)
    right_xy = right_63[:, :, :2].reshape(arr.shape[0], 42)
    pose18_xy = pose_75[:, :18, :2].reshape(arr.shape[0], 36)

    sentence_120 = np.concatenate(
        [left_xy, right_xy, pose18_xy],
        axis=1
    ).astype(np.float32)

    if sentence_120.shape != (30, 120):
        raise ValueError(f"sentence_AI 입력 shape 오류: {sentence_120.shape}, expected (30, 120)")

    return sentence_120


def extract_degree_280_from_411d(sequence_411d):
    """
    degree_AI 보고서 기준 입력:
    1F x 280D

    구성:
    16D 얼굴 요약 feature
    + 132D 정규화 얼굴 landmark
    + 132D delta feature
    = 280D

    현재 411D의 face 영역:
    face 201:411
    70점 x (x, y, confidence) = 210D

    이 중 앞 66개 얼굴점의 x, y를 사용해 132D를 만든다.
    """
    arr = np.asarray(sequence_411d, dtype=np.float32)

    if arr.ndim != 2 or arr.shape[1] != 411:
        raise ValueError(f"411D sequence shape 오류: {arr.shape}")

    face = arr[:, 201:411].reshape(arr.shape[0], 70, 3)
    face_xy = face[:, :66, :2]

    face_conf = face[:, :, 2]
    has_face = bool(np.any(face_conf > 0))

    # 대표 프레임: 중앙 프레임
    mid_idx = len(face_xy) // 2
    current_xy = face_xy[mid_idx]

    # 기준 프레임: 첫 프레임
    base_xy = face_xy[0]

    # 현재 프레임 정규화
    center = np.mean(current_xy, axis=0)
    centered = current_xy - center

    scale = np.std(centered)
    if scale < 1e-6:
        scale = 1.0

    norm_xy = centered / scale
    norm_132 = norm_xy.reshape(-1)

    # 기준 프레임 정규화
    base_center = np.mean(base_xy, axis=0)
    base_centered = base_xy - base_center

    base_scale = np.std(base_centered)
    if base_scale < 1e-6:
        base_scale = 1.0

    base_norm = base_centered / base_scale

    # delta feature
    delta_xy = norm_xy - base_norm
    delta_132 = delta_xy.reshape(-1)

    # 16D summary feature
    x = norm_xy[:, 0]
    y = norm_xy[:, 1]
    dx = delta_xy[:, 0]
    dy = delta_xy[:, 1]

    summary_16 = np.array([
        np.mean(x),
        np.std(x),
        np.min(x),
        np.max(x),
        np.mean(y),
        np.std(y),
        np.min(y),
        np.max(y),
        np.mean(dx),
        np.std(dx),
        np.min(dx),
        np.max(dx),
        np.mean(dy),
        np.std(dy),
        np.min(dy),
        np.max(dy),
    ], dtype=np.float32)

    degree_280 = np.concatenate([
        summary_16,
        norm_132.astype(np.float32),
        delta_132.astype(np.float32),
    ]).astype(np.float32)

    if degree_280.shape[0] != 280:
        raise ValueError(f"degree_AI 입력 shape 오류: {degree_280.shape}, expected (280,)")

    return degree_280, has_face

def _extract_with_external_mediapipe_worker(video_path: str, target_frames: int = 30):
    project_backend = Path(__file__).resolve().parents[2]
    worker_python = project_backend / ".venv_mp" / "Scripts" / "python.exe"
    worker_script = project_backend / "app" / "services" / "mp_extract_worker.py"

    if not worker_python.exists():
        raise FileNotFoundError(f"MediaPipe worker python not found: {worker_python}")

    if not worker_script.exists():
        raise FileNotFoundError(f"MediaPipe worker script not found: {worker_script}")

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        output_path = tmp.name

    try:
        completed = subprocess.run(
            [str(worker_python), str(worker_script), video_path, output_path],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                f"MediaPipe worker failed\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )

        arr = np.load(output_path).astype(np.float32)

        if arr.shape != (target_frames, 411):
            raise ValueError(f"worker output shape error: {arr.shape}")

        return arr.tolist()

    finally:
        try:
            os.remove(output_path)
        except OSError:
            pass