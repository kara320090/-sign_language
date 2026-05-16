import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

# ==============================================================================
# 1. 경로 및 윈도우 하이퍼파라미터 설정
# ==============================================================================
KEYPOINT_JSON_DIR = os.path.abspath(r"D:\SWPJ-4\F_only\01_real_sen_keypoint_F")
MORPHEME_BASE_DIR = os.path.abspath(r"D:\SWPJ-4\F_only\01_real_sen_morpheme_F\morpheme")
OUTPUT_DIR = os.path.abspath(r"D:\SWPJ-4\최종_Window_데이터")

WINDOW_SIZE = 30  
STRIDE = 10       

# --- [CSV 기반 단어 사전 동적 생성] ---
clean_csv_path = os.path.abspath(r"D:\SWPJ-4\전처리 완료(수정 완)\used_videos_final_CLEANED.csv")
clean_df = pd.read_csv(clean_csv_path)

WORD_TO_IDX = {}
for _, row in clean_df.iterrows():
    try:
        word_key = str(row.iloc[1]).strip().replace(" ", "")
        idx_val = int(row.iloc[3])
        if word_key and pd.notna(word_key):
            WORD_TO_IDX[word_key] = idx_val
    except Exception:
        continue

print(f"★ CSV 기반 단어 사전 구축 완료 (총 {len(WORD_TO_IDX)}개 단어 등록)")

# --- [morpheme 정답지 사전 빌드] ---
print("\n🔍 형태소 정답지 사전을 빌드합니다...", end="", flush=True)
morpheme_path_map = {}
if os.path.exists(MORPHEME_BASE_DIR):
    for root, dirs, files in os.walk(MORPHEME_BASE_DIR):
        for f in files:
            if f.lower().endswith('_morpheme.json'):
                pure_name = f.upper().replace('_MORPHEME.JSON', '').strip()
                morpheme_path_map[pure_name] = os.path.abspath(os.path.join(root, f))
    print(f" 등록 완료. (총 {len(morpheme_path_map)}개 정면 정답지 세트 맵핑)")

# ==============================================================================
# 2. ⚡ 실제 비디오 폴더 위치 초고속 메모리 맵핑 (RAM 주소록)
# ==============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n⚡ 외장하드 과부하 방지를 위한 비디오 폴더 주소록 RAM 적재 시작...", end="", flush=True)
disk_folder_map = {}
if os.path.exists(KEYPOINT_JSON_DIR):
    for num_dir in os.listdir(KEYPOINT_JSON_DIR):
        num_path = os.path.join(KEYPOINT_JSON_DIR, num_dir)
        if os.path.isdir(num_path):
            for v_dir in os.listdir(num_path):
                disk_folder_map[v_dir.upper().strip()] = os.path.abspath(os.path.join(num_path, v_dir))
    print(" 완공! (RAM 주소록 적재 완료)")

target_uids = clean_df.iloc[:, 0].astype(str).str.strip().str.upper().unique()
remaining_uids = [uid if uid.endswith('_F') else f"{uid}_F" for uid in target_uids]
sanitized_target_uids = ["NIA_SL_SEN0001_REAL01_F"] + remaining_uids

# 💡 RAM 대기방 및 배치 카운터 세팅
FINAL_X_ACCUMULATOR = []
FINAL_Y_ACCUMULATOR = []
match_success_count = 0
batch_counter = 1

print(f"\n🚀 총 {len(sanitized_target_uids)}개 비디오 대상 [안전장치 탑재형 배치 적재] 전처리를 시작합니다...")
pbar = tqdm(total=len(sanitized_target_uids), desc="윈도우 데이터셋 구축률", unit="Videos")

for video_uid in sanitized_target_uids:
    pbar.update(1)
        
    morph_path = morpheme_path_map.get(video_uid)
    video_target_dir = disk_folder_map.get(video_uid)
    
    if not morph_path or not video_target_dir or not os.path.exists(video_target_dir):
        continue  
        
    try:
        frame_files = os.listdir(video_target_dir)
        frame_list = []
        
        for f in frame_files:
            if f.lower().endswith('_keypoints.json'):
                f_parts = f.split('_')
                try:
                    frame_idx = int(f_parts[-2])
                    frame_list.append((frame_idx, os.path.abspath(os.path.join(video_target_dir, f))))
                except ValueError:
                    continue
                    
        if not frame_list or len(frame_list) < WINDOW_SIZE:
            continue
            
        frame_list.sort(key=lambda x: x[0])
        all_frames_coords = []
        
        for f_idx, kp_path in frame_list:
            with open(kp_path, 'r', encoding='utf-8') as f:
                kp_data = json.load(f)
            
            coords = []
            if 'people' in kp_data:
                people_data = kp_data['people']
                if isinstance(people_data, list) and len(people_data) > 0:
                    target_person = people_data[0]
                elif isinstance(people_data, dict):
                    target_person = people_data
                else:
                    target_person = {}
                
                for key_name in ['face_keypoints_2d', 'pose_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
                    if key_name in target_person and target_person[key_name]:
                        coords.extend(target_person[key_name])
                        if len(coords) >= 120:
                            break
                            
            if not coords:
                if 'frames' in kp_data and kp_data['frames']:
                    coords = kp_data['frames'][0].get('coordinates', [])
                elif 'coordinates' in kp_data:
                    coords = kp_data['coordinates']
                
            if coords:
                if len(coords) < 120: coords = coords + [0.0] * (120 - len(coords))
                elif len(coords) > 120: coords = coords[:120]
                all_frames_coords.append(coords)
                
        if not all_frames_coords:
            continue
            
        all_frames_coords = np.array(all_frames_coords, dtype=np.float32)
        total_frames = all_frames_coords.shape[0]
        all_frames_coords = np.where(np.abs(all_frames_coords) > 10, 0, all_frames_coords)

        with open(morph_path, 'r', encoding='utf-8') as f:
            morph_data = json.load(f)
            
        meta_key = 'metaData' if 'metaData' in morph_data else 'metadata'
        total_duration = morph_data[meta_key]['duration']
        
        frame_labels = [None] * total_frames
        for item in morph_data.get('data', []):
            s_time = item.get('start')
            e_time = item.get('end')
            attrs = item.get('attributes', [{}])
            w_name = attrs[0].get('name') if attrs else None
            if w_name:
                w_name = w_name.strip().replace(" ", "")
            
            if s_time is not None and e_time is not None and w_name:
                start_frame_idx = max(0, int((s_time / total_duration) * total_frames) - 3)
                end_frame_idx = min(total_frames, int((e_time / total_duration) * total_frames) + 3)
                for idx in range(start_frame_idx, end_frame_idx):
                    frame_labels[idx] = w_name

        video_has_valid_window = False
        for start_f in range(0, total_frames - WINDOW_SIZE + 1, STRIDE):
            end_f = start_f + WINDOW_SIZE
            window_word_list = frame_labels[start_f:end_f]
            
            valid_words_in_window = [w for w in window_word_list if w is not None and w in WORD_TO_IDX]
            
            if valid_words_in_window:
                chosen_word = Counter(valid_words_in_window).most_common(1)[0][0]
                window_coords = all_frames_coords[start_f:end_f, :]
                
                FINAL_X_ACCUMULATOR.append(window_coords)
                FINAL_Y_ACCUMULATOR.append(WORD_TO_IDX[chosen_word])
                video_has_valid_window = True
                
        if video_has_valid_window:
            match_success_count += 1
            
            # 🚨 [안전장치 핵심 핵심] 윈도우 3,000개 누적 시 중간 백업 후 RAM 청소
            if len(FINAL_X_ACCUMULATOR) >= 3000:
                pbar.write(f"   ➔ 💾 [RAM 폭발 방지] 데이터 3,000개 도달. batch_{batch_counter} 파일 백업 후 RAM을 비웁니다.")
                np.save(os.path.join(OUTPUT_DIR, f"batch_{batch_counter}_X.npy"), np.array(FINAL_X_ACCUMULATOR, dtype=np.float32))
                np.save(os.path.join(OUTPUT_DIR, f"batch_{batch_counter}_y.npy"), np.array(FINAL_Y_ACCUMULATOR, dtype=np.int32))
                
                # RAM 주머니 완전 초기화 (재부팅 원천 차단)
                FINAL_X_ACCUMULATOR.clear()
                FINAL_Y_ACCUMULATOR.clear()
                batch_counter += 1
            
    except Exception:
        continue

pbar.close()

# ==============================================================================
# 3. 🏁 남은 자투리 최종 저장
# ==============================================================================
if FINAL_X_ACCUMULATOR:
    pbar.write(f"   ➔ 💾 [최종 마무리] 남은 마지막 자투리 배치 {batch_counter} 파일 저장 중...")
    np.save(os.path.join(OUTPUT_DIR, f"batch_{batch_counter}_X.npy"), np.array(FINAL_X_ACCUMULATOR, dtype=np.float32))
    np.save(os.path.join(OUTPUT_DIR, f"batch_{batch_counter}_y.npy"), np.array(FINAL_Y_ACCUMULATOR, dtype=np.int32))
    batch_counter += 1

print(f"\n✨ [전처리 완료] 모든 데이터가 안전하게 '배치 파일' 단위로 분할 저장되었습니다!")
