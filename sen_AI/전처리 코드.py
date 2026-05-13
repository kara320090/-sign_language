import os
import json
import csv
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. 글로벌 설정
# ==========================================
SEQ_LEN = 30           
CHUNK_SIZE = 1000      
FEATURE_DIM = 120 

def get_origin_and_scale(pose_landmarks):
    try:
        r_sh = np.array([pose_landmarks[6], pose_landmarks[7]])
        l_sh = np.array([pose_landmarks[15], pose_landmarks[16]])
        origin = (r_sh + l_sh) / 2
        scale = np.linalg.norm(r_sh - l_sh) + 1e-6
        return origin, scale
    except:
        return np.array([0.5, 0.5]), 1.0

def normalize_points(landmarks, origin, scale, num_points):
    try:
        pts = np.array(landmarks).reshape(-1, 3)[:num_points]
        coords = pts[:, :2]
        conf = pts[:, 2]
        norm_coords = (coords - origin) / scale
        norm_coords[conf <= 0] = 0
        return norm_coords.flatten()
    except:
        return np.zeros(num_points * 2)

def save_batch(video_dict, base_out_path, batch_idx, used_videos):
    if not video_dict: return batch_idx
    x_batch, y_batch = [], []
    for v_id, frames in video_dict.items():
        n = len(frames)
        if n >= SEQ_LEN:
            indices = np.linspace(0, n - 1, SEQ_LEN).astype(int)
            resampled = [frames[i] for i in indices]
        else:
            resampled = frames + [np.zeros(FEATURE_DIM)] * (SEQ_LEN - n)
        x_batch.append(np.array(resampled))
        y_batch.append(0)
        used_videos.append([v_id, "LABEL", n])

    np.save(os.path.join(base_out_path, f"X_batch_{batch_idx:03d}.npy"), np.array(x_batch, dtype=np.float32))
    np.save(os.path.join(base_out_path, f"y_batch_{batch_idx:03d}.npy"), np.array(y_batch, dtype=np.int64))
    print(f"\n💾 [저장 완료] 배치 {batch_idx}번 세이브 성공. 메모리 확보됨.")
    video_dict.clear()
    return batch_idx + 1

# ==========================================
# 2. 메인 실행 함수
# ==========================================
def run_streaming_preprocessing():
    BASE_RAW_PATH = r"C:\Users\wolah\Desktop\새 폴더\F_only"
    BASE_OUT_PATH = r"D:\SWPJ-4\전처리 완료"
    os.makedirs(BASE_OUT_PATH, exist_ok=True)

    video_dict = {}
    used_videos = []
    batch_idx = 1
    success_count = 0

    # 1. 파일 목록 수집
    print("🔍 [1/2] 전체 파일 목록을 스캔 중입니다 (5~10분 소요)...")
    all_files = []
    for root, _, files in os.walk(BASE_RAW_PATH):
        for f in files:
            if f.lower().endswith('.json'):
                all_files.append(os.path.join(root, f))
    
    if not all_files:
        print("❌ JSON 파일을 찾지 못했습니다.")
        return

    # 2. 데이터 처리
    print(f"🚀 [2/2] {len(all_files)}개 파일의 데이터 처리를 시작합니다.")
    for filepath in tqdm(all_files, desc="진행 중"):
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        if len(parts) < 3: continue
        video_id = "_".join(parts[:-2])

        try:
            with open(filepath, 'r', encoding='utf-8') as jfile:
                data = json.load(jfile)
            
            p_data = data.get('people')
            if not p_data: continue
            person = p_data[0] if isinstance(p_data, list) else p_data
            
            pose = person.get('pose_keypoints_2d', [])
            if not pose or len(pose) < 54: continue

            l_hand = person.get('hand_left_keypoints_2d', [])
            r_hand = person.get('hand_right_keypoints_2d', [])

            origin, scale = get_origin_and_scale(pose)
            f_pose = normalize_points(pose, origin, scale, 18)
            f_lh = normalize_points(l_hand, origin, scale, 21)
            f_rh = normalize_points(r_hand, origin, scale, 21)
            combined = np.concatenate([f_pose, f_lh, f_rh])

            if video_id not in video_dict:
                video_dict[video_id] = []
                success_count += 1
                # 1,000개 영상이 확보될 때마다 중간 저장
                if success_count % CHUNK_SIZE == 0:
                    batch_idx = save_batch(video_dict, BASE_OUT_PATH, batch_idx, used_videos)

            video_dict[video_id].append(combined)

        except:
            continue

    # 3. 마무리 저장
    if video_dict:
        save_batch(video_dict, BASE_OUT_PATH, batch_idx, used_videos)

    with open(os.path.join(BASE_OUT_PATH, "used_videos.csv"), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["video_uid", "label", "frame_count"])
        writer.writerows(used_videos)

    print(f"\n🎉 모든 공정 완료! 최종 결과물 확인: {BASE_OUT_PATH}")

if __name__ == "__main__":
    run_streaming_preprocessing()