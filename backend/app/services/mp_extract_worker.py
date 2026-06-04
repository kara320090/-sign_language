import sys
import cv2
import numpy as np
import mediapipe as mp


def landmarks_to_xyc(landmarks, target_count):
    if landmarks is None:
        return [0.0] * (target_count * 3)

    result = []

    for lm in list(landmarks.landmark)[:target_count]:
        visibility = getattr(lm, "visibility", 1.0)
        result.extend([float(lm.x), float(lm.y), float(visibility)])

    current_count = len(result) // 3

    if current_count < target_count:
        result.extend([0.0] * ((target_count - current_count) * 3))

    return result


def make_411d(results):
    """
    411D 구성:
    pose 25점 x 3 = 75
    left hand 21점 x 3 = 63
    right hand 21점 x 3 = 63
    face 70점 x 3 = 210
    total = 411
    """
    pose = landmarks_to_xyc(results.pose_landmarks, 25)
    left = landmarks_to_xyc(results.left_hand_landmarks, 21)
    right = landmarks_to_xyc(results.right_hand_landmarks, 21)
    face = landmarks_to_xyc(results.face_landmarks, 70)

    frame = pose + left + right + face

    if len(frame) != 411:
        raise ValueError(f"411D length error: {len(frame)}")

    return frame


def extract(video_path, output_path, target_frames=30):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("invalid frame count")

    frame_indices = np.linspace(0, total_frames - 1, target_frames).astype(int)
    frame_set = set(frame_indices.tolist())

    sequence = []

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if idx in frame_set:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            sequence.append(make_411d(results))

        idx += 1

    holistic.close()
    cap.release()

    if not sequence:
        raise ValueError("no frames extracted")

    while len(sequence) < target_frames:
        sequence.append(sequence[-1])

    arr = np.asarray(sequence[:target_frames], dtype=np.float32)

    if arr.shape != (target_frames, 411):
        raise ValueError(f"output shape error: {arr.shape}")

    np.save(output_path, arr)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError("usage: python mp_extract_worker.py <video_path> <output_path>")

    video_path = sys.argv[1]
    output_path = sys.argv[2]

    extract(video_path, output_path)
    print("ok")