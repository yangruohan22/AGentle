import time
import threading
from collections import deque
import joblib
import numpy as np
import pandas as pd
import mne
import json
import requests
import neurokit2 as nk
from scipy.stats import entropy
from scipy.ndimage import binary_closing
from pylsl import resolve_byprop, StreamInlet
import warnings
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

# ================= 0. 环境补丁与配置 =================
warnings.filterwarnings('ignore')
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'trapz'): np.trapz = np.trapezoid

WINDOW_SIZE_SEC = 60.0
STEP_SIZE_SEC = 5.0
BEST_THRESHOLD = 0.42

EEG_SFREQ = 1000
PHYSIO_SFREQ = 1000
ET_SFREQ = 1200

# ================= 1. 动态定位路径与加载配置 =================
sub_id = sys.argv[1] if len(sys.argv) > 1 else "Unknown"
print(f"🧠 正在为被试 {sub_id} 唤醒 A-Gentle AI 引擎...")

model_dir = os.path.join(current_dir, "..", "model")
backend_config_dir = os.path.join(current_dir, "..", "backend", "experiment_data", sub_id, "config")

# 加载模型
live_model = joblib.load(os.path.join(model_dir, 'agentle_lgbm_champion.pkl'))
live_scaler = joblib.load(os.path.join(model_dir, 'agentle_scaler.pkl'))
expected_features = joblib.load(os.path.join(model_dir, 'agentle_features.pkl'))

# 🚀 动态加载被试专属 ICA 拼图
try:
    fif_path = os.path.join(backend_config_dir, f"{sub_id}_baseline_ica.fif")
    json_path = os.path.join(backend_config_dir, f"{sub_id}_ica_config.json")

    base_ica = mne.preprocessing.read_ica(fif_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        manual_config = json.load(f)
        manual_excludes = manual_config.get('manual_excludes', [])
    print(f"✅ 成功加载 {sub_id} 的专属配置！封印索引: {manual_excludes}")
except Exception as e:
    print(f"⚠️ 未找到专属配置，将不执行 ICA 去噪。详情: {e}")
    base_ica = None
    manual_excludes = []

print(f"✅ 系统就绪！要求对齐 {len(expected_features)} 个特征。")


# ================= 2. 实时预处理 =================
def preprocess_eeg_realtime(eeg_data_60s):
    ch_names = ['AF3', 'AF4', 'F3', 'F1', 'Fz', 'F2', 'F4', 'Pz']
    info = mne.create_info(ch_names=ch_names, sfreq=EEG_SFREQ, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data_60s, info, verbose=False)
    raw.notch_filter(freqs=50.0, fir_design='firwin', verbose=False)
    raw.filter(l_freq=1.0, h_freq=45.0, fir_design='firwin', verbose=False)

    if base_ica is not None:
        try:
            clean_raw = base_ica.apply(raw.copy(), exclude=manual_excludes, verbose=False)
            return clean_raw.get_data()
        except Exception as e:
            print(f"⚠️ ICA 应用异常: {e}")
            return raw.get_data()
    return raw.get_data()


# ================= 3. 核心特征提取 =================
def get_eeg_features_full_stream(eeg_raw):
    feat = {}
    try:
        eeg = eeg_raw * 1e6
        psds, freqs = mne.time_frequency.psd_array_welch(eeg, sfreq=1000, fmin=1.0, fmax=45, n_fft=1024, verbose=False)
        bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 45)}
        tot_p = psds.sum(axis=1)
        for name, (f1, f2) in bands.items():
            mask = (freqs >= f1) & (freqs <= f2)
            abs_p = psds[:, mask].sum(axis=1)
            for ch in range(8):
                feat[f'EEG_Ch{ch + 1}_{name}_Abs'] = abs_p[ch]
                feat[f'EEG_Ch{ch + 1}_{name}_Rel'] = abs_p[ch] / (tot_p[ch] + 1e-9)

        frontal_theta = psds[:2, (freqs >= 4) & (freqs <= 8)].sum()
        frontal_beta = psds[:2, (freqs >= 13) & (freqs <= 30)].sum()
        feat['EEG_Frontal_TBR'] = frontal_theta / (frontal_beta + 1e-9)
        feat['EEG_FAA_AF4_AF3'] = np.log(feat['EEG_Ch1_Alpha_Abs'] + 1e-9) - np.log(feat['EEG_Ch2_Alpha_Abs'] + 1e-9)

        for ch in range(8):
            try:
                feat[f'EEG_Ch{ch + 1}_ApproxEn'] = nk.entropy_approximate(eeg[ch][:20000])[0]
            except:
                feat[f'EEG_Ch{ch + 1}_ApproxEn'] = np.nan
            feat[f'EEG_Ch{ch + 1}_Fractal'] = nk.fractal_petrosian(eeg[ch])[0]
    except Exception as e:
        print(f" [EEG ] ❌ 报错: {e}")
    return feat


def get_ecg_features_robust_stream(ecg_data):
    feat = {f'ECG_HRV_{k}': np.nan for k in ['MeanNN', 'SDNN', 'RMSSD', 'pNN50', 'LF', 'HF', 'LFHF', 'SD1', 'SD2']}
    try:
        valid_ecg = ecg_data[~np.isnan(ecg_data)]
        if len(valid_ecg) < 1000: return feat
        cleaned = nk.ecg_clean(valid_ecg, sampling_rate=1000)
        peaks, info = nk.ecg_peaks(cleaned, sampling_rate=1000)
        peak_count = len(info['ECG_R_Peaks'])
        if 40 <= peak_count <= 150:
            hrv_df = nk.hrv(peaks, sampling_rate=1000)
            res = hrv_df.to_dict('records')[0]
            for k in ['MeanNN', 'SDNN', 'RMSSD', 'pNN50', 'LF', 'HF', 'LFHF', 'SD1', 'SD2']:
                feat[f'ECG_HRV_{k}'] = res.get(f'HRV_{k}', np.nan)
    except Exception as e:
        pass
    return feat


def get_gsr_features_stream_optimized(gsr_data):
    feat = {f'GSR_{k}': np.nan for k in ['Raw_Mean', 'SCL_Mean', 'SCR_Peaks', 'SCR_Amplitude', 'Phasic_STD']}
    try:
        valid_gsr = gsr_data[~np.isnan(gsr_data)]
        if len(valid_gsr) < 5000: return feat
        signals, info = nk.eda_process(valid_gsr, sampling_rate=1000, method='neurokit')
        feat['GSR_Raw_Mean'] = np.mean(valid_gsr)
        feat['GSR_SCL_Mean'] = signals['EDA_Tonic'].mean()
        feat['GSR_Phasic_STD'] = np.std(signals['EDA_Phasic'])
        feat['GSR_SCR_Peaks'] = len(info['SCR_Peaks'])
        if feat['GSR_SCR_Peaks'] > 0:
            feat['GSR_SCR_Amplitude'] = signals['EDA_Phasic'].iloc[info['SCR_Peaks']].mean()
        else:
            feat['GSR_SCR_Amplitude'] = 0.0
    except Exception as e:
        pass
    return feat


def get_et_features_enhanced(et):
    feat = {k: np.nan for k in ['ET_Pupil_Mean', 'ET_Gaze_Entropy', 'ET_Fix_Rate', 'ET_Blink_Count', 'ET_Saccade_Dist']}
    try:
        gx, gy = et[:, 1], et[:, 2]
        is_out_of_range = (gx <= 0.05) | (gx >= 0.95) | (gy <= 0.05) | (gy >= 0.95) | np.isnan(gx)
        g_vel = np.zeros_like(gx)
        g_vel[1:] = np.sqrt(np.diff(gx) ** 2 + np.diff(gy) ** 2)
        is_speed_burst = g_vel > 0.02
        blink_suspect = is_out_of_range | is_speed_burst
        merge_kernel = np.ones(60)
        merged_suspect = binary_closing(blink_suspect, structure=merge_kernel)
        events = nk.events_find(merged_suspect, threshold=0.5)
        blink_count = sum(1 for d in events.get('duration', []) if 60 <= d <= 720)
        feat['ET_Blink_Count'] = blink_count
        valid_mask = ~merged_suspect
        v_gx, v_gy = gx[valid_mask], gy[valid_mask]
        diff_mask = valid_mask[1:] & valid_mask[:-1]
        real_diffs = np.sqrt(np.diff(gx) ** 2 + np.diff(gy) ** 2)[diff_mask]
        if len(v_gx) > 500:
            moves = real_diffs[real_diffs > 1e-5]
            if len(moves) > 0: feat['ET_Saccade_Dist'] = np.mean(moves)
        hist, _, _ = np.histogram2d(v_gx, v_gy, bins=10, range=[[0, 1], [0, 1]])
        feat['ET_Gaze_Entropy'] = entropy(hist.flatten())
        feat['ET_Fix_Rate'] = (np.sum(real_diffs < 0.002) / len(gx)) * 100
        if et.shape[1] >= 5 and not np.all(np.isnan(et[:, 3])):
            feat['ET_Pupil_Mean'] = np.nanmean(et[valid_mask, 3:5])
    except Exception as e:
        pass
    return feat


# ================= 4. 在线对齐与推断 =================
def prepare_and_predict(features_dict):
    aligned_vector = []
    for feat_name in expected_features:
        if feat_name not in features_dict:
            aligned_vector.append(0.0)  # 容错
            continue
        val = features_dict[feat_name]
        aligned_vector.append(0.0 if np.isnan(val) else val)

    X_raw = np.array([aligned_vector])
    X_scaled = live_scaler.transform(X_raw)
    prob_state_1 = live_model.predict_proba(X_scaled)[0][1]
    return prob_state_1


# ================= 5. LSL 三路并发主循环 (多线程环形缓冲重构版) =================
def start_online_inference():
    print("\n📡 正在局域网内寻找数据流...")

    inlet_eeg = StreamInlet(resolve_byprop('name', 'Neuracle_EEG')[0])
    inlet_physio = StreamInlet(resolve_byprop('name', 'Physio_NI6009')[0])
    inlet_et = StreamInlet(resolve_byprop('name', 'EyeTracker')[0])
    print(f"✅ 成功锁定所有 LSL 流！准备收集第一个 60 秒...")

    # 计算各路流 60 秒所需的点数
    eeg_win = int(WINDOW_SIZE_SEC * EEG_SFREQ)
    physio_win = int(WINDOW_SIZE_SEC * PHYSIO_SFREQ)
    et_win = int(WINDOW_SIZE_SEC * ET_SFREQ)

    # 🚀 使用双端队列 (deque) 构建固定长度的环形缓冲区，老数据自动溢出抛弃
    eeg_buf = deque(maxlen=eeg_win)
    physio_buf = deque(maxlen=physio_win)
    et_buf = deque(maxlen=et_win)

    # 🧵 独立后台线程：无阻碍疯狂抓取最新数据
    def pull_data_worker():
        while True:
            e_chunk, _ = inlet_eeg.pull_chunk(timeout=0.0)
            p_chunk, _ = inlet_physio.pull_chunk(timeout=0.0)
            et_chunk, _ = inlet_et.pull_chunk(timeout=0.0)

            if e_chunk: eeg_buf.extend(e_chunk)
            if p_chunk: physio_buf.extend(p_chunk)
            if et_chunk: et_buf.extend(et_chunk)

            # 极短的休眠，防止拉满单核 CPU
            time.sleep(0.005)

    threading.Thread(target=pull_data_worker, daemon=True).start()

    print(f"⏳ 正在向水池注水，请等待 {WINDOW_SIZE_SEC} 秒...")
    # 阻塞主线程，同时打印实时注水进度！
    while len(eeg_buf) < eeg_win or len(physio_buf) < physio_win or len(et_buf) < et_win:
        # \r 可以让终端在同一行刷新，不会满屏乱滚
        print(
            f"💧 进度监测 -> 脑电: {len(eeg_buf)}/{eeg_win} | 生理: {len(physio_buf)}/{physio_win} | 眼动: {len(et_buf)}/{et_win}      ",
            end='\r')
        time.sleep(1)
    print("\n🚀 缓冲池已满！心流探测引擎正式启动...")

    # ⏱️ 严格计时的决策主循环
    while True:
        cycle_start = time.time()

        # 1. 瞬间抓取缓冲池当前快照 (转为 array)
        current_eeg_raw = np.array(eeg_buf).T[:8, :]
        current_physio = np.array(physio_buf).T
        current_et = np.array(et_buf)

        # 2. 预处理 (应用手动 ICA)
        cleaned_eeg = preprocess_eeg_realtime(current_eeg_raw)

        # 3. 提取特征
        all_features = {}
        all_features.update(get_eeg_features_full_stream(cleaned_eeg))
        all_features.update(get_gsr_features_stream_optimized(current_physio[0]))
        all_features.update(get_ecg_features_robust_stream(current_physio[1]))
        all_features.update(get_et_features_enhanced(current_et))

        # 4. 推断
        flow_prob = prepare_and_predict(all_features)

        process_time = time.time() - cycle_start

        if flow_prob >= BEST_THRESHOLD:
            print(f"[{time.strftime('%H:%M:%S')}] 🟢 心流状态 (Prob: {flow_prob:.2f}) | 计算耗时: {process_time:.2f}s")
        else:
            print(
                f"[{time.strftime('%H:%M:%S')}] 🔴 认知枯竭 (Prob: {flow_prob:.2f}) -> 触发干预报警！ | 计算耗时: {process_time:.2f}s")
            # 🔥 发送报警至后端，加上 timeout 防止后端卡死阻塞推断引擎
            try:
                requests.post("http://127.0.0.1:8000/api/alert_depletion", timeout=1.0)
            except:
                pass

        # 5. 动态计算补时休眠，严格锁死 5 秒步长！
        sleep_time = max(0.0, STEP_SIZE_SEC - process_time)
        time.sleep(sleep_time)


if __name__ == '__main__':
    start_online_inference()