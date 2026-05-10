import argparse
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from degree_features import normalize_face_points, extract_degree_features_from_points


LABEL_NAMES = ["weak", "normal", "strong"]

DEGREE_KO = {
    "weak": "약함",
    "normal": "보통",
    "strong": "강함",
}

MODIFIER_KO = {
    "weak": "조금",
    "normal": "",
    "strong": "매우",
}


def extract_frame_number(path: Path) -> int:
    """
    파일명에서 프레임 번호 추출.
    예:
    xxx_000000000240_keypoints.json
    frame_000123.json
    """
    nums = re.findall(r"\d+", path.stem)
    if not nums:
        return -1
    return int(nums[-1])


def recursive_find_face_keypoints(obj: Any) -> list[float] | None:
    """
    JSON 내부 어디에 있든 face_keypoints_2d를 찾는다.
    """
    if isinstance(obj, dict):
        if "face_keypoints_2d" in obj:
            return obj["face_keypoints_2d"]

        for value in obj.values():
            found = recursive_find_face_keypoints(value)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for item in obj:
            found = recursive_find_face_keypoints(item)
            if found is not None:
                return found

    return None


def get_face_keypoints_from_json(data: dict[str, Any]) -> list[float] | None:
    """
    OpenPose/AIHub 계열 JSON에서 face_keypoints_2d를 찾는다.

    지원 형태:
    1) {"face_keypoints_2d": [...]}
    2) {"people": [{"face_keypoints_2d": [...]}]}
    3) {"people": {"face_keypoints_2d": [...]}}
    4) 그 외 중첩 구조
    """
    if "face_keypoints_2d" in data:
        return data["face_keypoints_2d"]

    people = data.get("people")

    if isinstance(people, list) and len(people) > 0:
        best_face = None
        best_conf = -1.0

        for person in people:
            if not isinstance(person, dict):
                continue

            face = person.get("face_keypoints_2d")
            if face is None:
                continue

            arr = np.asarray(face, dtype=np.float32)

            if len(arr) % 3 == 0:
                conf = float(np.mean(arr.reshape(-1, 3)[:, 2]))
            else:
                conf = 1.0

            if conf > best_conf:
                best_conf = conf
                best_face = face

        if best_face is not None:
            return best_face

    if isinstance(people, dict):
        face = people.get("face_keypoints_2d")
        if face is not None:
            return face

    return recursive_find_face_keypoints(data)


def load_aihub_face_points(json_path: Path) -> tuple[np.ndarray, float]:
    """
    AIHub keypoint JSON에서 얼굴 landmark 좌표와 평균 confidence 반환.

    반환:
    points: (N, 2)
    mean_confidence: float
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    face = get_face_keypoints_from_json(data)

    if face is None:
        raise KeyError(f"face_keypoints_2d not found: {json_path}")

    arr = np.asarray(face, dtype=np.float32).flatten()

    if len(arr) == 0:
        raise ValueError(f"empty face_keypoints_2d: {json_path}")

    # OpenPose 형식: x, y, confidence 반복
    if len(arr) % 3 == 0:
        arr3 = arr.reshape(-1, 3)
        points = arr3[:, :2]
        confidence = arr3[:, 2]
        mean_conf = float(np.mean(confidence))

    # 일부 데이터가 x, y만 있을 경우
    elif len(arr) % 2 == 0:
        arr2 = arr.reshape(-1, 2)
        points = arr2
        mean_conf = 1.0

    else:
        raise ValueError(
            f"face_keypoints_2d length must be divisible by 3 or 2: "
            f"{json_path}, length={len(arr)}"
        )

    return points.astype(np.float32), mean_conf


def make_280_frame_feature(
    points: np.ndarray,
    prev_norm_flat: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    1F × 280D feature 생성.

    학습 때와 동일한 구성:
    16D summary feature
    + 132D normalized landmark
    + 132D previous-frame delta
    = 280D
    """
    summary_16 = extract_degree_features_from_points(points)

    norm_points = normalize_face_points(points)
    norm_flat = norm_points.flatten().astype(np.float32)

    # DISFA 학습은 66점 = 132차원 기준.
    # AIHub/OpenPose face가 70점이면 앞 66점만 사용.
    if len(norm_flat) > 132:
        norm_flat = norm_flat[:132]

    # 66점보다 적으면 padding
    if len(norm_flat) < 132:
        padded = np.zeros(132, dtype=np.float32)
        padded[: len(norm_flat)] = norm_flat
        norm_flat = padded

    if prev_norm_flat is None:
        delta = np.zeros_like(norm_flat, dtype=np.float32)
    else:
        delta = norm_flat - prev_norm_flat

    feature = np.concatenate([summary_16, norm_flat, delta], axis=0).astype(np.float32)

    if feature.shape[0] != 280:
        raise ValueError(f"Expected 280 feature dims, got {feature.shape[0]}")

    return feature, norm_flat


def load_model_bundle(model_path: Path) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    bundle = joblib.load(model_path)

    if isinstance(bundle, dict) and "model" in bundle:
        return bundle

    return {
        "model": bundle,
        "label_names": LABEL_NAMES,
        "input_shape": [280],
        "requires_flatten": False,
    }


def find_sequence_dirs(input_path: Path) -> list[Path]:
    """
    input_path가 keypoint JSON들을 직접 포함하면 그 폴더 하나 반환.
    아니면 하위 폴더 중 keypoint JSON을 가진 폴더들을 모두 반환.
    """
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if input_path.is_file():
        raise ValueError("input_path must be a directory.")

    direct_jsons = list(input_path.glob("*.json"))
    direct_keypoints = [
        p for p in direct_jsons
        if "keypoint" in p.name.lower() or "keypoints" in p.name.lower()
    ]

    if direct_keypoints:
        return [input_path]

    sequence_dirs = []

    for p in input_path.rglob("*"):
        if not p.is_dir():
            continue

        jsons = list(p.glob("*.json"))
        keypoints = [
            j for j in jsons
            if "keypoint" in j.name.lower() or "keypoints" in j.name.lower()
        ]

        if keypoints:
            sequence_dirs.append(p)

    return sorted(sequence_dirs)


def find_keypoint_jsons(sequence_dir: Path) -> list[Path]:
    jsons = list(sequence_dir.glob("*.json"))

    keypoints = [
        p for p in jsons
        if "keypoint" in p.name.lower() or "keypoints" in p.name.lower()
    ]

    if keypoints:
        return sorted(keypoints, key=extract_frame_number)

    return sorted(jsons, key=extract_frame_number)


def safe_sequence_name(root: Path, seq_dir: Path) -> str:
    try:
        rel = seq_dir.relative_to(root)
        name = "__".join(rel.parts)
    except Exception:
        name = seq_dir.name

    name = name.replace(":", "").replace("\\", "__").replace("/", "__")
    return name if name else seq_dir.name


def predict_sequence_1f_280d(
    bundle: dict,
    sequence_dir: Path,
    min_face_confidence: float = 0.05,
) -> tuple[dict, pd.DataFrame]:
    model = bundle["model"]
    label_names = bundle.get("label_names", LABEL_NAMES)

    json_files = find_keypoint_jsons(sequence_dir)

    if not json_files:
        raise FileNotFoundError(f"No json files found in {sequence_dir}")

    rows = []
    probs_list = []
    prev_norm_flat = None

    for json_path in json_files:
        frame_no = extract_frame_number(json_path)

        try:
            points, face_conf = load_aihub_face_points(json_path)

            if face_conf < min_face_confidence:
                rows.append({
                    "frame": frame_no,
                    "file": str(json_path),
                    "status": "skip_low_face_confidence",
                    "face_confidence": face_conf,
                    "degree": "",
                    "confidence": np.nan,
                })
                continue

            feature, prev_norm_flat = make_280_frame_feature(points, prev_norm_flat)
            x = feature.reshape(1, -1)

            pred_idx = int(model.predict(x)[0])

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(x)[0]
            else:
                probs = np.zeros(len(label_names), dtype=np.float32)
                probs[pred_idx] = 1.0

            probs_list.append(probs)

            row = {
                "frame": frame_no,
                "file": str(json_path),
                "status": "ok",
                "face_confidence": face_conf,
                "degree": label_names[pred_idx],
                "degree_ko": DEGREE_KO.get(label_names[pred_idx], label_names[pred_idx]),
                "confidence": float(probs[pred_idx]),
            }

            for i, name in enumerate(label_names):
                row[f"prob_{name}"] = float(probs[i])

            rows.append(row)

        except Exception as e:
            rows.append({
                "frame": frame_no,
                "file": str(json_path),
                "status": "error",
                "error": str(e),
                "degree": "",
                "confidence": np.nan,
            })

    frame_df = pd.DataFrame(rows)

    if not probs_list:
        result = {
            "sequence_dir": str(sequence_dir),
            "model_input": "1F_x_280D",
            "degree": "normal",
            "degree_ko": "보통",
            "confidence": 0.0,
            "probs": {
                "weak": 0.0,
                "normal": 1.0,
                "strong": 0.0,
            },
            "num_total_frames": len(json_files),
            "num_used_frames": 0,
            "warning": "No valid face frames were used.",
        }
        return result, frame_df

    probs_avg = np.mean(np.stack(probs_list, axis=0), axis=0)
    pred_idx = int(np.argmax(probs_avg))
    degree = label_names[pred_idx]

    result = {
        "sequence_dir": str(sequence_dir),
        "model_input": "1F_x_280D",
        "degree": degree,
        "degree_ko": DEGREE_KO.get(degree, degree),
        "confidence": float(probs_avg[pred_idx]),
        "probs": {
            label_names[i]: float(probs_avg[i])
            for i in range(len(label_names))
        },
        "num_total_frames": len(json_files),
        "num_used_frames": len(probs_list),
    }

    return result, frame_df


def make_final_text(sign_text: str, degree: str) -> str:
    if not sign_text:
        return ""

    modifier = MODIFIER_KO.get(degree, "")

    if modifier == "":
        return sign_text

    return f"{modifier} {sign_text}"


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="models/degree_frame_280d_anger_mlp.joblib",
        help="최종 degree_AI 모델 경로",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="AIHub keypoint JSON 폴더 또는 상위 폴더",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/degree_aihub_1f280_mlp",
    )
    parser.add_argument(
        "--min_face_confidence",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=0,
        help="0이면 전체 처리",
    )
    parser.add_argument(
        "--sign_text",
        type=str,
        default="",
        help="예: 화난다, 슬프다 등. 입력하면 degree와 결합한 final_text를 저장",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)

    bundle = load_model_bundle(model_path)

    sequence_dirs = find_sequence_dirs(input_path)

    if args.max_sequences > 0:
        sequence_dirs = sequence_dirs[: args.max_sequences]

    if not sequence_dirs:
        raise RuntimeError(f"No keypoint sequence folders found under {input_path}")

    print(f"[INFO] model: {model_path}")
    print(f"[INFO] input: {input_path}")
    print(f"[INFO] found sequence dirs: {len(sequence_dirs)}")
    print(f"[INFO] out_dir: {out_dir}")

    summary_rows = []

    for seq_dir in tqdm(sequence_dirs, desc="Applying degree_AI"):
        try:
            result, frame_df = predict_sequence_1f_280d(
                bundle=bundle,
                sequence_dir=seq_dir,
                min_face_confidence=args.min_face_confidence,
            )

            if args.sign_text:
                result["sign_text"] = args.sign_text
                result["final_text"] = make_final_text(
                    sign_text=args.sign_text,
                    degree=result["degree"],
                )

            seq_name = safe_sequence_name(input_path, seq_dir)
            seq_out_dir = out_dir / seq_name
            seq_out_dir.mkdir(parents=True, exist_ok=True)

            frame_csv = seq_out_dir / "degree_frame_predictions.csv"
            result_json = seq_out_dir / "degree_result.json"

            frame_df.to_csv(frame_csv, index=False, encoding="utf-8-sig")
            save_json(result_json, result)

            row = {
                "sequence_name": seq_name,
                "sequence_dir": str(seq_dir),
                "model_path": str(model_path),
                "model_input": result.get("model_input"),
                "degree": result.get("degree"),
                "degree_ko": result.get("degree_ko"),
                "confidence": result.get("confidence"),
                "prob_weak": result.get("probs", {}).get("weak"),
                "prob_normal": result.get("probs", {}).get("normal"),
                "prob_strong": result.get("probs", {}).get("strong"),
                "num_total_frames": result.get("num_total_frames"),
                "num_used_frames": result.get("num_used_frames"),
                "result_json": str(result_json),
                "frame_csv": str(frame_csv),
                "status": "ok",
            }

            if args.sign_text:
                row["sign_text"] = args.sign_text
                row["final_text"] = result.get("final_text")

            summary_rows.append(row)

        except Exception as e:
            summary_rows.append({
                "sequence_name": seq_dir.name,
                "sequence_dir": str(seq_dir),
                "model_path": str(model_path),
                "status": "error",
                "error": str(e),
            })

    summary_df = pd.DataFrame(summary_rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "degree_aihub_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n[DONE] AIHub degree_AI application finished.")
    print(f"Summary saved to: {summary_path}")

    if "degree" in summary_df.columns:
        print("\nDegree counts:")
        print(summary_df["degree"].value_counts(dropna=False))

    if "status" in summary_df.columns:
        print("\nStatus counts:")
        print(summary_df["status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()