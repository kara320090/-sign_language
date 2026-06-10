import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np


BACKEND_ROOT = Path(__file__).resolve().parents[2]

POSE_POINT_COUNT = 25
HAND_POINT_COUNT = 21
FACE_POINT_COUNT = 70

POSE_LEN = POSE_POINT_COUNT * 3
HAND_LEN = HAND_POINT_COUNT * 3
FACE_LEN = FACE_POINT_COUNT * 3
WORD_FEATURE_DIM = POSE_LEN + HAND_LEN + HAND_LEN + FACE_LEN
SENTENCE_FEATURE_DIM = 120
DEGREE_FEATURE_DIM = 280

_live_holistic = None
_live_holistic_options = None
_live_holistic_lock = threading.Lock()
_live_component_landmarkers = None
_live_component_options = None
_live_component_lock = threading.Lock()


BODY_25_FROM_MEDIAPIPE = [
    0,  # 0 Nose
    (11, 12),  # 1 Neck
    12,  # 2 RShoulder
    14,  # 3 RElbow
    16,  # 4 RWrist
    11,  # 5 LShoulder
    13,  # 6 LElbow
    15,  # 7 LWrist
    (23, 24),  # 8 MidHip
    24,  # 9 RHip
    26,  # 10 RKnee
    28,  # 11 RAnkle
    23,  # 12 LHip
    25,  # 13 LKnee
    27,  # 14 LAnkle
    5,  # 15 REye
    2,  # 16 LEye
    8,  # 17 REar
    7,  # 18 LEar
    31,  # 19 LBigToe
    31,  # 20 LSmallToe
    29,  # 21 LHeel
    32,  # 22 RBigToe
    32,  # 23 RSmallToe
    30,  # 24 RHeel
]

FACE_70_FROM_MEDIAPIPE = [
    127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400,
    378, 379, 365, 70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 168,
    6, 197, 195, 5, 4, 45, 220, 115, 275, 440, 344, 33, 160, 158, 133,
    153, 144, 362, 385, 387, 263, 373, 380, 61, 146, 91, 181, 84, 17,
    314, 405, 321, 375, 291, 308, 78, 95, 88, 178, 87, 14,
]


def _landmark_visibility(landmark: Any) -> float:
    visibility = getattr(landmark, "visibility", None)
    presence = getattr(landmark, "presence", None)
    values = [
        float(value)
        for value in (visibility, presence)
        if value is not None and np.isfinite(float(value))
    ]
    if not values:
        return 1.0
    return float(min(values))


def _mediapipe_landmarks(landmarks: Any) -> list[Any]:
    if landmarks is None:
        return []

    legacy_landmarks = getattr(landmarks, "landmark", None)
    if legacy_landmarks is not None:
        return list(legacy_landmarks)

    try:
        return list(landmarks)
    except TypeError:
        return []


def _mediapipe_point(landmarks: Any, index: int, width: int, height: int) -> list[float]:
    items = _mediapipe_landmarks(landmarks)
    if index < 0 or index >= len(items):
        return [0.0, 0.0, 0.0]

    landmark = items[index]
    x = float(landmark.x) * width
    y = float(landmark.y) * height
    confidence = _landmark_visibility(landmark)

    if not np.isfinite(x) or not np.isfinite(y) or confidence <= 0:
        return [0.0, 0.0, 0.0]
    return [x, y, confidence]


def _mediapipe_average_point(
    landmarks: Any,
    indices: tuple[int, int],
    width: int,
    height: int,
) -> list[float]:
    first = _mediapipe_point(landmarks, indices[0], width, height)
    second = _mediapipe_point(landmarks, indices[1], width, height)
    confidence = min(first[2], second[2])
    if confidence <= 0:
        return [0.0, 0.0, 0.0]
    return [
        (first[0] + second[0]) / 2.0,
        (first[1] + second[1]) / 2.0,
        confidence,
    ]


def _flatten_points(points: Sequence[Sequence[float]]) -> list[float]:
    return [float(value) for point in points for value in point]


def _mediapipe_pose_keypoints(landmarks: Any, width: int, height: int) -> list[float]:
    points = []
    for source in BODY_25_FROM_MEDIAPIPE:
        if isinstance(source, tuple):
            points.append(_mediapipe_average_point(landmarks, source, width, height))
        else:
            points.append(_mediapipe_point(landmarks, source, width, height))
    return _flatten_points(points)


def _mediapipe_hand_keypoints(landmarks: Any, width: int, height: int) -> list[float]:
    points = [
        _mediapipe_point(landmarks, index, width, height)
        for index in range(HAND_POINT_COUNT)
    ]
    return _flatten_points(points)


def _mediapipe_face_keypoints(landmarks: Any, width: int, height: int) -> list[float]:
    points = [
        _mediapipe_point(landmarks, index, width, height)
        for index in FACE_70_FROM_MEDIAPIPE
    ]

    if len(points) >= 52:
        right_eye_center = np.mean(np.asarray(points[39:45], dtype=np.float32), axis=0)
        left_eye_center = np.mean(np.asarray(points[45:51], dtype=np.float32), axis=0)
        points.extend([right_eye_center.tolist(), left_eye_center.tolist()])

    if len(points) < FACE_POINT_COUNT:
        points.extend([[0.0, 0.0, 0.0]] * (FACE_POINT_COUNT - len(points)))

    return _flatten_points(points[:FACE_POINT_COUNT])


def _mediapipe_frame_to_keypoint_schema(results: Any, width: int, height: int) -> dict[str, Any]:
    person = {
        "person_id": [-1],
        "pose_keypoints_2d": _mediapipe_pose_keypoints(
            results.pose_landmarks, width, height
        ),
        "face_keypoints_2d": _mediapipe_face_keypoints(
            results.face_landmarks, width, height
        ),
        "hand_left_keypoints_2d": _mediapipe_hand_keypoints(
            results.left_hand_landmarks, width, height
        ),
        "hand_right_keypoints_2d": _mediapipe_hand_keypoints(
            results.right_hand_landmarks, width, height
        ),
        "pose_keypoints_3d": [],
        "face_keypoints_3d": [],
        "hand_left_keypoints_3d": [],
        "hand_right_keypoints_3d": [],
    }

    return {
        "version": "mediapipe-holistic",
        "people": [person],
    }


def _mediapipe_task_model_path() -> Path:
    configured = os.environ.get("MEDIAPIPE_HOLISTIC_TASK_PATH") or os.environ.get(
        "MEDIAPIPE_TASK_PATH"
    )
    if configured:
        model_path = Path(configured).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"MEDIAPIPE_HOLISTIC_TASK_PATH does not exist: {model_path}"
            )
        return model_path.resolve()

    candidates = [
        BACKEND_ROOT / "models" / "mediapipe" / "holistic_landmarker.task",
        BACKEND_ROOT / "mediapipe" / "holistic_landmarker.task",
        BACKEND_ROOT.parent / "models" / "mediapipe" / "holistic_landmarker.task",
        BACKEND_ROOT.parent / "모델" / "mediapipe" / "holistic_landmarker.task",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "MediaPipe 0.10.33+ requires holistic_landmarker.task. "
        "Set MEDIAPIPE_HOLISTIC_TASK_PATH to the downloaded task model path."
    )


def _mediapipe_component_task_model_path(
    env_name: str,
    filename: str,
    description: str,
) -> Path:
    configured = os.environ.get(env_name)
    if configured:
        model_path = Path(configured).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"{env_name} does not exist: {model_path}")
        return model_path.resolve()

    candidates = [
        BACKEND_ROOT / "models" / "mediapipe" / filename,
        BACKEND_ROOT / "mediapipe" / filename,
        BACKEND_ROOT.parent / "models" / "mediapipe" / filename,
        BACKEND_ROOT.parent / "모델" / "mediapipe" / filename,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"MediaPipe {description} task model is required. "
        f"Set {env_name} to the downloaded {filename} path."
    )


def _first_landmarks(groups: Any) -> Any:
    if groups is None:
        return None
    try:
        return groups[0] if groups else None
    except (IndexError, TypeError):
        return None


def _hand_label(handedness: Any) -> str:
    if not handedness:
        return ""
    first = handedness[0] if isinstance(handedness, list) else handedness
    return str(
        getattr(first, "category_name", None)
        or getattr(first, "display_name", None)
        or getattr(first, "categoryName", "")
    ).lower()


class _AdjustedLandmark:
    def __init__(self, source: Any, x: float, y: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(getattr(source, "z", 0.0))
        self.visibility = getattr(source, "visibility", None)
        self.presence = getattr(source, "presence", None)


class _FaceResult:
    def __init__(self, face_landmarks: Any):
        self.face_landmarks = [face_landmarks] if face_landmarks else []


def _face_result_has_landmarks(face_result: Any) -> bool:
    return bool(_first_landmarks(getattr(face_result, "face_landmarks", None)))


def _upper_body_crop_box(
    pose_result: Any,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    pose_landmarks = _first_landmarks(getattr(pose_result, "pose_landmarks", None))
    landmarks = _mediapipe_landmarks(pose_landmarks)
    upper_body_indices = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16,
    ]
    points = []

    for index in upper_body_indices:
        if index >= len(landmarks):
            continue

        landmark = landmarks[index]
        confidence = _landmark_visibility(landmark)
        x = float(getattr(landmark, "x", 0.0)) * width
        y = float(getattr(landmark, "y", 0.0)) * height
        if confidence <= 0.05 or not np.isfinite(x) or not np.isfinite(y):
            continue
        if x < -0.1 * width or x > 1.1 * width:
            continue
        if y < -0.1 * height or y > 1.1 * height:
            continue
        points.append((x, y))

    if len(points) < 2:
        return (
            int(width * 0.25),
            0,
            int(width * 0.75),
            int(height * 0.75),
        )

    array = np.asarray(points, dtype=np.float32)
    x_min, y_min = np.min(array, axis=0)
    x_max, y_max = np.max(array, axis=0)
    box_width = max(float(x_max - x_min), width * 0.12)
    box_height = max(float(y_max - y_min), height * 0.18)

    pad_x = max(box_width * 0.35, width * 0.04)
    pad_top = max(box_height * 0.45, height * 0.08)
    pad_bottom = max(box_height * 0.25, height * 0.04)

    x1 = max(0, int(x_min - pad_x))
    y1 = max(0, int(y_min - pad_top))
    x2 = min(width, int(x_max + pad_x))
    y2 = min(height, int(y_max + pad_bottom))

    if x2 - x1 < width * 0.18 or y2 - y1 < height * 0.18:
        return (
            int(width * 0.25),
            0,
            int(width * 0.75),
            int(height * 0.75),
        )

    return x1, y1, x2, y2


def _detect_face_on_upper_body_crop(
    face_landmarker: Any,
    frame: Any,
    cv2: Any,
    mp: Any,
    pose_result: Any,
    width: int,
    height: int,
) -> Any:
    x1, y1, x2, y2 = _upper_body_crop_box(pose_result, width, height)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return _FaceResult(None)

    crop_height, crop_width = crop.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    crop_result = face_landmarker.detect(image)
    crop_landmarks = _first_landmarks(
        getattr(crop_result, "face_landmarks", None)
    )
    if not crop_landmarks:
        return crop_result

    adjusted_landmarks = [
        _AdjustedLandmark(
            landmark,
            (x1 + float(landmark.x) * crop_width) / width,
            (y1 + float(landmark.y) * crop_height) / height,
        )
        for landmark in _mediapipe_landmarks(crop_landmarks)
    ]
    return _FaceResult(adjusted_landmarks)


def _detect_face_with_crop_fallback(
    face_landmarker: Any,
    frame: Any,
    cv2: Any,
    mp: Any,
    pose_result: Any,
    width: int,
    height: int,
) -> Any:
    crop_result = _detect_face_on_upper_body_crop(
        face_landmarker,
        frame,
        cv2,
        mp,
        pose_result,
        width,
        height,
    )
    if _face_result_has_landmarks(crop_result):
        return crop_result

    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return face_landmarker.detect(image)


def _combined_component_results(
    pose_result: Any,
    hand_result: Any,
    face_result: Any,
) -> Any:
    left_hand_landmarks = None
    right_hand_landmarks = None

    hand_landmarks = getattr(hand_result, "hand_landmarks", []) or []
    handedness = getattr(hand_result, "handedness", []) or []
    for index, landmarks in enumerate(hand_landmarks):
        label = _hand_label(handedness[index] if index < len(handedness) else [])
        if label == "left" and left_hand_landmarks is None:
            left_hand_landmarks = landmarks
        elif label == "right" and right_hand_landmarks is None:
            right_hand_landmarks = landmarks
        elif left_hand_landmarks is None:
            left_hand_landmarks = landmarks
        elif right_hand_landmarks is None:
            right_hand_landmarks = landmarks

    class ComponentResults:
        pass

    combined = ComponentResults()
    combined.pose_landmarks = _first_landmarks(
        getattr(pose_result, "pose_landmarks", None)
    )
    combined.face_landmarks = _first_landmarks(
        getattr(face_result, "face_landmarks", None)
    )
    combined.left_hand_landmarks = left_hand_landmarks
    combined.right_hand_landmarks = right_hand_landmarks
    return combined


def _run_mediapipe_solutions(
    capture: Any,
    cv2: Any,
    holistic_module: Any,
    frame_step: int,
    model_complexity: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    static_image_mode: bool = False,
) -> list[dict[str, Any]]:
    frames = []
    frame_index = 0

    with holistic_module.Holistic(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as holistic:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % frame_step != 0:
                frame_index += 1
                continue

            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            frames.append(_mediapipe_frame_to_keypoint_schema(results, width, height))
            frame_index += 1

    return frames


def _run_mediapipe_holistic_task(
    capture: Any,
    cv2: Any,
    mp: Any,
    frame_step: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> list[dict[str, Any]]:
    model_path = _mediapipe_task_model_path()
    vision = mp.tasks.vision
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    options = vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        min_face_detection_confidence=min_detection_confidence,
        min_face_landmarks_confidence=min_tracking_confidence,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_landmarks_confidence=min_tracking_confidence,
        min_hand_landmarks_confidence=min_tracking_confidence,
        output_face_blendshapes=False,
        output_segmentation_mask=False,
    )

    frames = []
    frame_index = 0
    landmarker = vision.HolisticLandmarker.create_from_options(options)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % frame_step != 0:
                frame_index += 1
                continue

            height, width = frame.shape[:2]
            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_index * 1000.0 / fps)
            results = landmarker.detect_for_video(image, timestamp_ms)
            frames.append(_mediapipe_frame_to_keypoint_schema(results, width, height))
            frame_index += 1
    finally:
        landmarker.close()

    return frames


def _run_mediapipe_component_tasks(
    capture: Any,
    cv2: Any,
    mp: Any,
    frame_step: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> list[dict[str, Any]]:
    pose_model_path = _mediapipe_component_task_model_path(
        "MEDIAPIPE_POSE_TASK_PATH",
        "pose_landmarker_full.task",
        "pose",
    )
    hand_model_path = _mediapipe_component_task_model_path(
        "MEDIAPIPE_HAND_TASK_PATH",
        "hand_landmarker.task",
        "hand",
    )
    face_model_path = _mediapipe_component_task_model_path(
        "MEDIAPIPE_FACE_TASK_PATH",
        "face_landmarker.task",
        "face",
    )

    vision = mp.tasks.vision
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    pose_options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(pose_model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_tracking_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=False,
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_tracking_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(face_model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=min_detection_confidence,
        min_face_presence_confidence=min_tracking_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    frames = []
    frame_index = 0
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
    face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % frame_step != 0:
                frame_index += 1
                continue

            height, width = frame.shape[:2]
            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_index * 1000.0 / fps)

            pose_result = pose_landmarker.detect_for_video(image, timestamp_ms)
            hand_result = hand_landmarker.detect_for_video(image, timestamp_ms)
            face_result = _detect_face_with_crop_fallback(
                face_landmarker,
                frame,
                cv2,
                mp,
                pose_result,
                width,
                height,
            )
            combined = _combined_component_results(
                pose_result,
                hand_result,
                face_result,
            )
            frames.append(_mediapipe_frame_to_keypoint_schema(combined, width, height))
            frame_index += 1
    finally:
        pose_landmarker.close()
        hand_landmarker.close()
        face_landmarker.close()

    return frames


def _run_mediapipe_tasks(
    capture: Any,
    cv2: Any,
    mp: Any,
    frame_step: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> list[dict[str, Any]]:
    prefer_holistic = os.environ.get("MEDIAPIPE_PREFER_HOLISTIC_TASK") == "1"

    if not prefer_holistic:
        try:
            return _run_mediapipe_component_tasks(
                capture,
                cv2,
                mp,
                frame_step,
                min_detection_confidence,
                min_tracking_confidence,
            )
        except FileNotFoundError:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except Exception as component_error:
            raise RuntimeError(
                f"MediaPipe component task extraction failed: {component_error}"
            ) from component_error

    try:
        return _run_mediapipe_holistic_task(
            capture,
            cv2,
            mp,
            frame_step,
            min_detection_confidence,
            min_tracking_confidence,
        )
    except Exception as holistic_error:
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return _run_mediapipe_component_tasks(
                capture,
                cv2,
                mp,
                frame_step,
                min_detection_confidence,
                min_tracking_confidence,
            )
        except Exception as component_error:
            raise RuntimeError(
                "MediaPipe holistic task failed and component task fallback "
                f"also failed. holistic={holistic_error}; "
                f"components={component_error}"
            ) from component_error


class _SingleFrameCapture:
    def __init__(self, frame: Any):
        self._frame = frame
        self._read = False

    def read(self) -> tuple[bool, Any]:
        if self._read:
            return False, None
        self._read = True
        return True, self._frame

    def get(self, prop_id: int) -> float:
        return 30.0

    def set(self, prop_id: int, value: float) -> bool:
        if value <= 0:
            self._read = False
        return True


def _mediapipe_options() -> tuple[int, float, float]:
    model_complexity = int(os.environ.get("MEDIAPIPE_MODEL_COMPLEXITY", "1"))
    min_detection_confidence = float(
        os.environ.get("MEDIAPIPE_MIN_DETECTION_CONFIDENCE", "0.5")
    )
    min_tracking_confidence = float(
        os.environ.get("MEDIAPIPE_MIN_TRACKING_CONFIDENCE", "0.5")
    )
    return model_complexity, min_detection_confidence, min_tracking_confidence


def _get_live_holistic(
    holistic_module: Any,
    model_complexity: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> Any:
    global _live_holistic, _live_holistic_options

    options = (
        model_complexity,
        min_detection_confidence,
        min_tracking_confidence,
    )
    if _live_holistic is not None and _live_holistic_options == options:
        return _live_holistic

    if _live_holistic is not None:
        _live_holistic.close()

    _live_holistic = holistic_module.Holistic(
        static_image_mode=True,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    _live_holistic_options = options
    return _live_holistic


def _get_live_component_landmarkers(
    mp: Any,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> tuple[Any, Any, Any]:
    global _live_component_landmarkers, _live_component_options

    pose_model_path = _mediapipe_component_task_model_path(
        "MEDIAPIPE_POSE_TASK_PATH",
        "pose_landmarker_full.task",
        "pose",
    )
    hand_model_path = _mediapipe_component_task_model_path(
        "MEDIAPIPE_HAND_TASK_PATH",
        "hand_landmarker.task",
        "hand",
    )
    face_model_path = _mediapipe_component_task_model_path(
        "MEDIAPIPE_FACE_TASK_PATH",
        "face_landmarker.task",
        "face",
    )

    options = (
        str(pose_model_path),
        str(hand_model_path),
        str(face_model_path),
        min_detection_confidence,
        min_tracking_confidence,
    )
    if (
        _live_component_landmarkers is not None
        and _live_component_options == options
    ):
        return _live_component_landmarkers

    if _live_component_landmarkers is not None:
        for landmarker in _live_component_landmarkers:
            landmarker.close()

    vision = mp.tasks.vision
    pose_options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(pose_model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_tracking_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=False,
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_tracking_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(face_model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=min_detection_confidence,
        min_face_presence_confidence=min_tracking_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    _live_component_landmarkers = (
        vision.PoseLandmarker.create_from_options(pose_options),
        vision.HandLandmarker.create_from_options(hand_options),
        vision.FaceLandmarker.create_from_options(face_options),
    )
    _live_component_options = options
    return _live_component_landmarkers


def _run_mediapipe_on_frame(frame: Any, cv2: Any, mp: Any) -> dict[str, Any]:
    model_complexity, min_detection_confidence, min_tracking_confidence = (
        _mediapipe_options()
    )

    holistic_module = getattr(getattr(mp, "solutions", None), "holistic", None)
    if holistic_module is not None:
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        with _live_holistic_lock:
            holistic = _get_live_holistic(
                holistic_module,
                model_complexity,
                min_detection_confidence,
                min_tracking_confidence,
            )
            results = holistic.process(rgb)
        return _mediapipe_frame_to_keypoint_schema(results, width, height)
    else:
        height, width = frame.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with _live_component_lock:
            pose_landmarker, hand_landmarker, face_landmarker = (
                _get_live_component_landmarkers(
                    mp,
                    min_detection_confidence,
                    min_tracking_confidence,
                )
            )
            pose_result = pose_landmarker.detect(image)
            hand_result = hand_landmarker.detect(image)
            face_result = _detect_face_with_crop_fallback(
                face_landmarker,
                frame,
                cv2,
                mp,
                pose_result,
                width,
                height,
            )

        combined = _combined_component_results(
            pose_result,
            hand_result,
            face_result,
        )
        return _mediapipe_frame_to_keypoint_schema(combined, width, height)


def _run_mediapipe(video_path: str) -> tuple[dict[str, Any], ...]:
    video = Path(video_path).resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video}")

    try:
        import cv2
        import mediapipe as mp
    except ImportError as error:
        raise RuntimeError(
            "MediaPipe keypoint extraction requires mediapipe and opencv-python."
        ) from error

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video for MediaPipe: {video}")

    frame_step = max(1, int(os.environ.get("MEDIAPIPE_FRAME_STEP", "1")))
    model_complexity, min_detection_confidence, min_tracking_confidence = (
        _mediapipe_options()
    )

    try:
        holistic_module = getattr(getattr(mp, "solutions", None), "holistic", None)
        if holistic_module is not None:
            frames = _run_mediapipe_solutions(
                capture,
                cv2,
                holistic_module,
                frame_step,
                model_complexity,
                min_detection_confidence,
                min_tracking_confidence,
            )
        else:
            frames = _run_mediapipe_tasks(
                capture,
                cv2,
                mp,
                frame_step,
                min_detection_confidence,
                min_tracking_confidence,
            )
    finally:
        capture.release()

    if not frames:
        raise RuntimeError("No valid MediaPipe keypoint frames were produced.")

    return tuple(frames)


@lru_cache(maxsize=16)
def extract_mediapipe_frames_from_video(video_path: str) -> tuple[dict[str, Any], ...]:
    return _run_mediapipe(video_path)


def extract_mediapipe_frame_from_image_bytes(content: bytes) -> dict[str, Any]:
    if not content:
        raise ValueError("Image content is empty.")

    try:
        import cv2
        import mediapipe as mp
    except ImportError as error:
        raise RuntimeError(
            "MediaPipe keypoint extraction requires mediapipe and opencv-python."
        ) from error

    encoded = np.frombuffer(content, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode image content.")

    return _run_mediapipe_on_frame(frame, cv2, mp)


def _get_people(data: dict[str, Any]) -> list[dict[str, Any]]:
    people = data.get("people")
    if isinstance(people, list):
        return [person for person in people if isinstance(person, dict)]
    if isinstance(people, dict):
        return [people]
    return []


def _word_person(data: dict[str, Any]) -> dict[str, Any]:
    people = data.get("people", {})
    if isinstance(people, list):
        if not people:
            return {}
        return people[0] if isinstance(people[0], dict) else {}
    if isinstance(people, dict):
        return people
    return {}


def _fix_length(values: Any, target_len: int) -> list[float]:
    if not isinstance(values, list):
        return [0.0] * target_len
    if len(values) >= target_len:
        return values[:target_len]
    return values + [0.0] * (target_len - len(values))


def _reshape_fixed_keypoints(values: Any, target_len: int) -> np.ndarray:
    return np.asarray(_fix_length(values, target_len), dtype=np.float32).reshape(-1, 3)


def _word_origin_and_scale(pose: np.ndarray) -> tuple[np.ndarray, float]:
    origin = np.array([0.0, 0.0], dtype=np.float32)
    scale = 1.0

    try:
        neck = pose[1]
        right_shoulder = pose[2]
        left_shoulder = pose[5]

        if right_shoulder[2] > 0 and left_shoulder[2] > 0:
            origin = (right_shoulder[:2] + left_shoulder[:2]) / 2.0
            shoulder_width = float(
                np.linalg.norm(right_shoulder[:2] - left_shoulder[:2])
            )
            if shoulder_width > 1e-6:
                scale = shoulder_width
        elif neck[2] > 0:
            origin = neck[:2].astype(np.float32)
    except Exception:
        pass

    return origin, scale


def _normalize_word_points(
    points: np.ndarray,
    origin: np.ndarray,
    scale: float,
) -> np.ndarray:
    normalized = points.copy()
    confident = normalized[:, 2] > 0
    normalized[~confident, 0:2] = 0.0
    normalized[confident, 0] = (normalized[confident, 0] - origin[0]) / scale
    normalized[confident, 1] = (normalized[confident, 1] - origin[1]) / scale
    return normalized


def _word_frame_feature(data: dict[str, Any]) -> np.ndarray:
    person = _word_person(data)
    pose = _reshape_fixed_keypoints(person.get("pose_keypoints_2d", []), POSE_LEN)
    left = _reshape_fixed_keypoints(
        person.get("hand_left_keypoints_2d", []), HAND_LEN
    )
    right = _reshape_fixed_keypoints(
        person.get("hand_right_keypoints_2d", []), HAND_LEN
    )
    face = _reshape_fixed_keypoints(person.get("face_keypoints_2d", []), FACE_LEN)

    origin, scale = _word_origin_and_scale(pose)
    features = np.concatenate(
        [
            _normalize_word_points(pose, origin, scale).ravel(),
            _normalize_word_points(left, origin, scale).ravel(),
            _normalize_word_points(right, origin, scale).ravel(),
            _normalize_word_points(face, origin, scale).ravel(),
        ]
    ).astype(np.float32)

    if features.shape != (WORD_FEATURE_DIM,):
        raise ValueError(
            f"Word frame shape mismatch: {features.shape}, "
            f"expected ({WORD_FEATURE_DIM},)"
        )

    return features


def _resize_word_sequence(sequence: np.ndarray, target_frames: int) -> np.ndarray:
    current_len = len(sequence)
    if current_len == 0:
        raise ValueError("MediaPipe sequence is empty.")
    if current_len == target_frames:
        return sequence.astype(np.float32)
    if current_len > target_frames:
        indices = np.linspace(0, current_len - 1, target_frames).astype(int)
        return sequence[indices].astype(np.float32)

    padding = np.zeros(
        (target_frames - current_len, sequence.shape[1]), dtype=np.float32
    )
    return np.vstack([sequence, padding]).astype(np.float32)


def preprocess_word_mediapipe_frames(
    frames: Sequence[dict[str, Any]],
    target_frames: int = 30,
) -> np.ndarray:
    raw = np.stack([_word_frame_feature(frame) for frame in frames], axis=0)
    return _resize_word_sequence(raw, target_frames)


def extract_word_411d_sequence_from_video(
    video_path: str,
    target_frames: int = 30,
) -> np.ndarray:
    frames = extract_mediapipe_frames_from_video(video_path)
    return preprocess_word_mediapipe_frames(frames, target_frames)


def extract_411d_sequence_from_video(
    video_path: str,
    target_frames: int = 30,
) -> np.ndarray:
    return extract_word_411d_sequence_from_video(video_path, target_frames)


def extract_hands_126_from_411d(sequence: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != WORD_FEATURE_DIM:
        raise ValueError(f"411D sequence shape mismatch: {array.shape}")

    hands = array[:, 75:201].astype(np.float32)
    if hands.shape != (30, 126):
        raise ValueError(
            f"Word model input shape mismatch: {hands.shape}, expected (30, 126)"
        )
    return hands


def _mean_keypoint_confidence(values: Any) -> float | None:
    if values is None:
        return None

    array = np.asarray(values, dtype=np.float32).flatten()
    if len(array) == 0 or len(array) % 3 != 0:
        return None
    return float(np.mean(array.reshape(-1, 3)[:, 2]))


def _sentence_person(data: dict[str, Any]) -> dict[str, Any]:
    people = data.get("people")
    if isinstance(people, list) and people:
        best = people[0] if isinstance(people[0], dict) else {}
        best_score = -1.0

        for person in people:
            if not isinstance(person, dict):
                continue

            scores = []
            for key in (
                "hand_left_keypoints_2d",
                "hand_right_keypoints_2d",
                "pose_keypoints_2d",
            ):
                score = _mean_keypoint_confidence(person.get(key))
                if score is not None:
                    scores.append(score)

            person_score = float(np.mean(scores)) if scores else 0.0
            if person_score > best_score:
                best_score = person_score
                best = person

        return best

    if isinstance(people, dict):
        return people

    return data


def _recursive_find_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _recursive_find_key(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _recursive_find_key(item, key)
            if found is not None:
                return found
    return None


def _sentence_keypoints_to_xy(
    values: Any,
    num_points: int,
    width: float = 1920.0,
    height: float = 1080.0,
    confidence_threshold: float = 0.05,
) -> np.ndarray:
    output = np.zeros((num_points, 2), dtype=np.float32)
    if values is None:
        return output

    array = np.asarray(values, dtype=np.float32).flatten()
    if len(array) == 0:
        return output

    if len(array) % 3 == 0:
        array_3d = array.reshape(-1, 3)
        xy = array_3d[:, :2]
        confidence = array_3d[:, 2]
    elif len(array) % 2 == 0:
        xy = array.reshape(-1, 2)
        confidence = np.ones((len(xy),), dtype=np.float32)
    else:
        return output

    count = min(num_points, len(xy))
    xy = xy[:count].astype(np.float32)
    confidence = confidence[:count].astype(np.float32)

    if np.nanmax(xy) > 2.0:
        xy[:, 0] = xy[:, 0] / width
        xy[:, 1] = xy[:, 1] / height

    xy[confidence < confidence_threshold] = 0.0
    xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
    xy = np.clip(xy, 0.0, 1.0)
    output[:count] = xy
    return output


def _sentence_frame_feature(data: dict[str, Any]) -> np.ndarray:
    person = _sentence_person(data)

    left_values = person.get("hand_left_keypoints_2d")
    right_values = person.get("hand_right_keypoints_2d")
    pose_values = person.get("pose_keypoints_2d")

    if left_values is None:
        left_values = _recursive_find_key(data, "hand_left_keypoints_2d")
    if right_values is None:
        right_values = _recursive_find_key(data, "hand_right_keypoints_2d")
    if pose_values is None:
        pose_values = _recursive_find_key(data, "pose_keypoints_2d")

    left_xy = _sentence_keypoints_to_xy(left_values, HAND_POINT_COUNT)
    right_xy = _sentence_keypoints_to_xy(right_values, HAND_POINT_COUNT)
    pose_xy = _sentence_keypoints_to_xy(pose_values, POSE_POINT_COUNT)

    feature = np.concatenate(
        [left_xy.flatten(), right_xy.flatten(), pose_xy[:18].flatten()]
    ).astype(np.float32)

    if feature.shape != (SENTENCE_FEATURE_DIM,):
        raise ValueError(
            f"Sentence frame shape mismatch: {feature.shape}, "
            f"expected ({SENTENCE_FEATURE_DIM},)"
        )
    return feature


def _resize_sentence_sequence(sequence: np.ndarray, target_frames: int) -> np.ndarray:
    current_len = len(sequence)
    if current_len == 0:
        return np.zeros((target_frames, SENTENCE_FEATURE_DIM), dtype=np.float32)
    if current_len == target_frames:
        return sequence.astype(np.float32)
    if current_len > target_frames:
        indices = np.linspace(0, current_len - 1, target_frames).astype(int)
        return sequence[indices].astype(np.float32)

    padding = np.repeat(sequence[-1:], target_frames - current_len, axis=0)
    return np.concatenate([sequence, padding], axis=0).astype(np.float32)


def preprocess_sentence_mediapipe_frames(
    frames: Sequence[dict[str, Any]],
    target_frames: int = 30,
) -> np.ndarray:
    if not frames:
        return np.zeros((target_frames, SENTENCE_FEATURE_DIM), dtype=np.float32)

    raw = np.stack([_sentence_frame_feature(frame) for frame in frames], axis=0)
    return _resize_sentence_sequence(raw, target_frames)


def extract_sentence_120_sequence_from_video(
    video_path: str,
    target_frames: int = 30,
) -> np.ndarray:
    frames = extract_mediapipe_frames_from_video(video_path)
    return preprocess_sentence_mediapipe_frames(frames, target_frames)


def _safe_point(points: np.ndarray, index: int) -> np.ndarray | None:
    if index < 0 or index >= len(points):
        return None
    point = points[index]
    if not np.all(np.isfinite(point)):
        return None
    return point.astype(np.float32)


def _distance(points: np.ndarray, first: int, second: int) -> float:
    first_point = _safe_point(points, first)
    second_point = _safe_point(points, second)
    if first_point is None or second_point is None:
        return 0.0
    return float(np.linalg.norm(first_point - second_point))


def _mean_axis(points: np.ndarray, indices: list[int], axis: int) -> float:
    values = []
    for index in indices:
        point = _safe_point(points, index)
        if point is not None:
            values.append(float(point[axis]))
    return float(np.mean(values)) if values else 0.0


def _normalize_face_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Face points shape mismatch: {points.shape}")

    valid = np.isfinite(points).all(axis=1)
    if valid.sum() < 10:
        return np.zeros_like(points, dtype=np.float32)

    valid_points = points[valid]
    center = valid_points.mean(axis=0, keepdims=True)
    min_xy = valid_points.min(axis=0)
    max_xy = valid_points.max(axis=0)
    scale = float(max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]))
    if scale < 1e-6:
        scale = 1.0
    return ((points - center) / scale).astype(np.float32)


def _extract_degree_summary(points: np.ndarray) -> np.ndarray:
    points = _normalize_face_points(points)

    left_eye_width = _distance(points, 36, 39)
    left_eye_open = (
        _distance(points, 37, 41) + _distance(points, 38, 40)
    ) / 2.0
    right_eye_width = _distance(points, 42, 45)
    right_eye_open = (
        _distance(points, 43, 47) + _distance(points, 44, 46)
    ) / 2.0

    left_eye_ratio = left_eye_open / (left_eye_width + 1e-6)
    right_eye_ratio = right_eye_open / (right_eye_width + 1e-6)
    left_brow_y = _mean_axis(points, [17, 18, 19, 20, 21], 1)
    left_eye_y = _mean_axis(points, [36, 37, 38, 39, 40, 41], 1)
    right_brow_y = _mean_axis(points, [22, 23, 24, 25, 26], 1)
    right_eye_y = _mean_axis(points, [42, 43, 44, 45, 46, 47], 1)
    brow_eye_left = abs(left_eye_y - left_brow_y)
    brow_eye_right = abs(right_eye_y - right_brow_y)
    mouth_width = _distance(points, 48, 54)
    face_width = _distance(points, 0, 16)
    face_height = _distance(points, 8, 27)

    if face_width < 1e-6:
        face_width = 1.0
    if face_height < 1e-6:
        face_height = 1.0

    if len(points) > 66:
        mouth_open = (
            _distance(points, 62, 66) + _distance(points, 63, 65)
        ) / 2.0
    else:
        mouth_open = (
            _distance(points, 61, 64) + _distance(points, 62, 63)
        ) / 2.0

    x_span = float(np.nanmax(points[:, 0]) - np.nanmin(points[:, 0]))
    y_span = float(np.nanmax(points[:, 1]) - np.nanmin(points[:, 1]))

    summary = np.array(
        [
            left_eye_ratio,
            right_eye_ratio,
            abs(left_eye_ratio - right_eye_ratio),
            brow_eye_left,
            brow_eye_right,
            abs(brow_eye_left - brow_eye_right),
            mouth_width,
            mouth_open,
            mouth_open / (mouth_width + 1e-6),
            _mean_axis(points, [48, 54], 0),
            _mean_axis(points, [51, 57], 1),
            face_width,
            face_height,
            y_span / (x_span + 1e-6),
            x_span,
            y_span,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(summary, nan=0.0, posinf=0.0, neginf=0.0)


def _degree_face_values(data: dict[str, Any]) -> Any:
    if "face_keypoints_2d" in data:
        return data["face_keypoints_2d"]

    people = data.get("people")
    if isinstance(people, list) and people:
        best_face = None
        best_confidence = -1.0

        for person in people:
            if not isinstance(person, dict):
                continue
            face = person.get("face_keypoints_2d")
            if face is None:
                continue

            array = np.asarray(face, dtype=np.float32).flatten()
            if len(array) > 0 and len(array) % 3 == 0:
                confidence = float(np.mean(array.reshape(-1, 3)[:, 2]))
            else:
                confidence = 1.0

            if confidence > best_confidence:
                best_confidence = confidence
                best_face = face

        if best_face is not None:
            return best_face

    if isinstance(people, dict) and people.get("face_keypoints_2d") is not None:
        return people["face_keypoints_2d"]

    return _recursive_find_key(data, "face_keypoints_2d")


def _face_points_and_confidence(values: Any) -> tuple[np.ndarray, float]:
    array = np.asarray(values, dtype=np.float32).flatten()
    if len(array) == 0:
        raise ValueError("face_keypoints_2d is empty.")

    if len(array) % 3 == 0:
        array_3d = array.reshape(-1, 3)
        return array_3d[:, :2].astype(np.float32), float(np.mean(array_3d[:, 2]))
    if len(array) % 2 == 0:
        return array.reshape(-1, 2).astype(np.float32), 1.0

    raise ValueError(
        "face_keypoints_2d length must be divisible by 3 or 2, "
        f"got {len(array)}"
    )


def _degree_frame_feature(
    points: np.ndarray,
    previous_normalized: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    summary = _extract_degree_summary(points)
    normalized = _normalize_face_points(points).flatten().astype(np.float32)

    if len(normalized) > 132:
        normalized = normalized[:132]
    if len(normalized) < 132:
        padded = np.zeros(132, dtype=np.float32)
        padded[: len(normalized)] = normalized
        normalized = padded

    delta = (
        np.zeros_like(normalized, dtype=np.float32)
        if previous_normalized is None
        else normalized - previous_normalized
    )
    feature = np.concatenate([summary, normalized, delta]).astype(np.float32)

    if feature.shape != (DEGREE_FEATURE_DIM,):
        raise ValueError(
            f"Degree frame shape mismatch: {feature.shape}, "
            f"expected ({DEGREE_FEATURE_DIM},)"
        )
    return feature, normalized


def preprocess_degree_mediapipe_frames(
    frames: Sequence[dict[str, Any]],
    min_face_confidence: float = 0.05,
) -> tuple[np.ndarray, bool]:
    features = []
    previous_normalized = None

    for frame in frames:
        face_values = _degree_face_values(frame)
        if face_values is None:
            continue

        try:
            points, face_confidence = _face_points_and_confidence(face_values)
            if face_confidence < min_face_confidence:
                continue

            feature, previous_normalized = _degree_frame_feature(
                points, previous_normalized
            )
            features.append(feature)
        except (TypeError, ValueError):
            continue

    if not features:
        return np.zeros((0, DEGREE_FEATURE_DIM), dtype=np.float32), False

    return np.stack(features, axis=0).astype(np.float32), True


def extract_degree_280_sequence_from_video(
    video_path: str,
    min_face_confidence: float = 0.05,
) -> tuple[np.ndarray, bool]:
    frames = extract_mediapipe_frames_from_video(video_path)
    return preprocess_degree_mediapipe_frames(frames, min_face_confidence)


def _keypoints_detected(values: Any) -> bool:
    if values is None:
        return False

    array = np.asarray(values, dtype=np.float32).flatten()
    if len(array) == 0:
        return False
    if len(array) % 3 == 0:
        return bool(np.any(array.reshape(-1, 3)[:, 2] > 0))
    return bool(np.any(array))


def summarize_mediapipe_frames(
    frames: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    has_pose = False
    has_left_hand = False
    has_right_hand = False
    has_face = False

    for frame in frames:
        people = _get_people(frame)
        if not people and isinstance(frame, dict):
            people = [frame]

        for person in people:
            has_pose = has_pose or _keypoints_detected(
                person.get("pose_keypoints_2d")
            )
            has_left_hand = has_left_hand or _keypoints_detected(
                person.get("hand_left_keypoints_2d")
            )
            has_right_hand = has_right_hand or _keypoints_detected(
                person.get("hand_right_keypoints_2d")
            )
            has_face = has_face or _keypoints_detected(
                person.get("face_keypoints_2d")
            )

    return {
        "extractor": "mediapipe",
        "sequence_length": int(len(frames)),
        "frame_dim": WORD_FEATURE_DIM,
        "has_pose": has_pose,
        "has_left_hand": has_left_hand,
        "has_right_hand": has_right_hand,
        "has_face": has_face,
    }
