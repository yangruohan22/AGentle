import sys
import os
import json
import time
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
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
        # 🌟 2秒护航逻辑 A：必须先等录制脚本把 .npy 写完！
        eeg_npy_path = os.path.join(config_dir, f"{sub_id}_baseline_eeg.npy")
        print("⏳ 正在等待原始录制数据落盘...")
        while not os.path.exists(eeg_npy_path):
            time.sleep(2.0)

        # 1. 读取录制下来的 3 分钟原始数据
        eeg_raw = np.load(eeg_npy_path).T
        bio_raw = np.load(os.path.join(config_dir, f"{sub_id}_baseline_bio.npy"))
        et_raw = np.load(os.path.join(config_dir, f"{sub_id}_baseline_et.npy"))

        eeg_sfreq, physio_sfreq, et_sfreq = 1000, 1000, 1200
        chunks_features = []

        for i in range(3):
            eeg_chunk = eeg_raw[:, i * 60 * eeg_sfreq: (i + 1) * 60 * eeg_sfreq]
            bio_chunk = bio_raw[i * 60 * physio_sfreq: (i + 1) * 60 * physio_sfreq, :]
            et_chunk = et_raw[i * 60 * et_sfreq: (i + 1) * 60 * et_sfreq, :]

            if eeg_chunk.shape[1] < 60 * eeg_sfreq * 0.9:
                continue

            feat = {}
            feat.update(get_eeg_features_full_stream(eeg_chunk))
            feat.update(get_gsr_features_stream_optimized(bio_chunk[:, 0]))
            feat.update(get_ecg_features_robust_stream(bio_chunk[:, 1]))
            feat.update(get_et_features_enhanced(et_chunk))
            chunks_features.append(feat)

        if not chunks_features:
            raise ValueError("没有足够的数据切片来计算基线。")

        df_features = pd.DataFrame(chunks_features)
        mean_features = df_features.mean(skipna=True).to_dict()
        safe_dict = {k: float(v) for k, v in mean_features.items() if not pd.isna(v)}

        # 🌟 2秒护航逻辑 B：等待兄弟进程的 ICA 图画完！
        ica_png_path = os.path.join(config_dir, f"{sub_id}_ica_panel.png")
        print("⏳ 特征均值计算完毕，正在耐心等待 ICA 地形图绘制完成...")
        while not os.path.exists(ica_png_path):
            time.sleep(2.0)  # 就是这个 2 秒查询！

        # 图画完了，才写 JSON！这样前端一看到 JSON，图必然已经存在，绝对不会 404！
        json_path = os.path.join(config_dir, f"{sub_id}_base_means.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(safe_dict, f, ensure_ascii=False, indent=4)

        print(f"✅ 基线特征提取完毕！共存入 {len(safe_dict)} 个均值特征至 {json_path}")
    except Exception as e:
        print(f"❌ 计算基线均值失败: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate_base_means(sys.argv[1])