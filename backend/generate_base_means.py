import sys
import os
import json
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加 online_system 路径，以便复用相同的特征提取逻辑
sys.path.append(os.path.join(current_dir, "..", "online_system"))

from main_inference import (
    get_eeg_features_full_stream,
    get_ecg_features_robust_stream,
    get_gsr_features_stream_optimized,
    get_et_features_enhanced
)


def generate_base_means(sub_id):
    print(f"🔄 正在为被试 {sub_id} 计算 3 分钟基线特征均值...")
    config_dir = os.path.join(current_dir, "experiment_data", sub_id, "config")

    try:
        # 1. 读取录制下来的 3 分钟原始数据
        eeg_raw = np.load(os.path.join(config_dir, f"{sub_id}_baseline_eeg.npy")).T
        bio_raw = np.load(os.path.join(config_dir, f"{sub_id}_baseline_bio.npy"))
        et_raw = np.load(os.path.join(config_dir, f"{sub_id}_baseline_et.npy"))

        eeg_sfreq, physio_sfreq, et_sfreq = 1000, 1000, 1200
        chunks_features = []

        # 2. 🌟 核心修复：将 3 分钟切成 3 个 60 秒的切片分别提取，避开时间窗不匹配导致的阈值拦截和能量偏移！
        for i in range(3):
            eeg_chunk = eeg_raw[:, i * 60 * eeg_sfreq: (i + 1) * 60 * eeg_sfreq]
            bio_chunk = bio_raw[i * 60 * physio_sfreq: (i + 1) * 60 * physio_sfreq, :]
            et_chunk = et_raw[i * 60 * et_sfreq: (i + 1) * 60 * et_sfreq, :]

            # 如果切片不够长 (比如最后一段没录满)，跳过
            if eeg_chunk.shape[1] < 60 * eeg_sfreq * 0.9:
                continue

            feat = {}
            feat.update(get_eeg_features_full_stream(eeg_chunk))
            # 🌟 修复索引越界：线上 LSL 流只有 2 个通道，0 是 GSR，1 是 ECG
            feat.update(get_gsr_features_stream_optimized(bio_chunk[:, 0]))
            feat.update(get_ecg_features_robust_stream(bio_chunk[:, 1]))
            feat.update(get_et_features_enhanced(et_chunk))

            chunks_features.append(feat)

        if not chunks_features:
            raise ValueError("没有足够的数据切片来计算基线。")

        # 3. 将 3 段的特征转换为 DataFrame 并对列求均值 (自动忽略里面的 NaN)
        df_features = pd.DataFrame(chunks_features)
        mean_features = df_features.mean(skipna=True).to_dict()

        # 4. 剔除彻底坏死的特征，存入 JSON
        safe_dict = {k: float(v) for k, v in mean_features.items() if not pd.isna(v)}

        json_path = os.path.join(config_dir, f"{sub_id}_base_means.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(safe_dict, f, ensure_ascii=False, indent=4)

        print(f"✅ 基线特征提取完毕！共存入 {len(safe_dict)} 个均值特征至 {json_path}")
    except Exception as e:
        print(f"❌ 计算基线均值失败: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate_base_means(sys.argv[1])