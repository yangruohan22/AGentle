import time
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

# ================= 1. 唤醒模型武器库与手动 ICA 配置 =================
print("🧠 正在唤醒 A-Gentle AI 引擎...")
live_model = joblib.load('agentle_lgbm_champion.pkl')
live_scaler = joblib.load('agentle_scaler.pkl')
expected_features = joblib.load('agentle_features.pkl')

# 🚀 加载手动 ICA 拼图
try:
    # 加载基线生成的 ICA 模型 (注意路径需指向 backend 目录)
    base_ica = mne.preprocessing.read_ica('../backend/agentle_baseline_ica.fif')
    # 加载前端保存的手动剔除黑名单
    with open('../backend/ica_config.json', 'r') as f:
        manual_config = json.load(f)
        manual_excludes = manual_config.get('manual_excludes', [])
    print(f"✅ 手动 ICA 配置加载成功！封印索引: {manual_excludes}")
except Exception as e:
    print(f"⚠️ ICA 配置加载失败 (可能是文件未生成)，将不执行 ICA 去噪: {e}")
    base_ica = None
    manual_excludes = []

print(f"✅ 系统就绪！要求对齐 {len(expected_features)} 个特征。")


# ================= 2. 实时预处理 (手动排雷版) =================
def preprocess_eeg_realtime(eeg_data_60s):
    """
    接收 60 秒数据，应用基线期的 ICA 矩阵和手动黑名单。
    """
    ch_names = ['AF3', 'AF4', 'F3', 'F1', 'Fz', 'F2', 'F4', 'Pz']
    info = mne.create_info(ch_names=ch_names, sfreq=EEG_SFREQ, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data_60s, info, verbose=False)

    # 基础滤波
    raw.notch_filter(freqs=50.0, fir_design='firwin', verbose=False)
    raw.filter(l_freq=1.0, h_freq=45.0, fir_design='firwin', verbose=False)

    # 🚀 应用手动排雷结果 (不再实时 fit)
    if base_ica is not None:
        try:
            clean_raw = base_ica.apply(raw.copy(), exclude=manual_excludes, verbose=False)
            return clean_raw.get_data()
        except Exception as e:
            print(f"⚠️ ICA 应用异常: {e}")
            return raw.get_data()
    return raw.get_data()


# ================= 3. 核心特征提取 (保持原样) =================

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


# ================= 5. LSL 三路并发主循环 =================
def start_online_inference():
    print("\n📡 正在局域网内寻找数据流...")

    # 修复：使用 resolve_byprop 解决 ImportError
    inlet_eeg = StreamInlet(resolve_byprop('name', 'Neuracle_EEG')[0])
    inlet_physio = StreamInlet(resolve_byprop('name', 'Physio_NI6009')[0])
    inlet_et = StreamInlet(resolve_byprop('name', 'EyeTracker')[0])
    print(f"✅ 成功锁定所有 LSL 流！准备收集第一个 60 秒...")

    eeg_buf, physio_buf, et_buf = [], [], []
    eeg_win, physio_win, et_win = int(WINDOW_SIZE_SEC * 1000), int(WINDOW_SIZE_SEC * 1000), int(WINDOW_SIZE_SEC * 1200)
    eeg_step, physio_step, et_step = int(STEP_SIZE_SEC * 1000), int(STEP_SIZE_SEC * 1000), int(STEP_SIZE_SEC * 1200)

    while True:
        e_chunk, _ = inlet_eeg.pull_chunk(timeout=0.0)
        p_chunk, _ = inlet_physio.pull_chunk(timeout=0.0)
        et_chunk, _ = inlet_et.pull_chunk(timeout=0.0)

        if e_chunk: eeg_buf.extend(e_chunk)
        if p_chunk: physio_buf.extend(p_chunk)
        if et_chunk: et_buf.extend(et_chunk)

        if len(eeg_buf) >= eeg_win and len(physio_buf) >= physio_win and len(et_buf) >= et_win:
            t_start = time.time()

            # 切片
            current_eeg_raw = np.array(eeg_buf[:eeg_win]).T[:8, :]
            current_physio = np.array(physio_buf[:physio_win]).T
            current_et = np.array(et_buf[:et_win])

            # 预处理 (应用手动 ICA)
            cleaned_eeg = preprocess_eeg_realtime(current_eeg_raw)

            # 提取特征
            all_features = {}
            all_features.update(get_eeg_features_full_stream(cleaned_eeg))
            all_features.update(get_gsr_features_stream_optimized(current_physio[0]))
            all_features.update(get_ecg_features_robust_stream(current_physio[1]))
            all_features.update(get_et_features_enhanced(current_et))

            # 推断
            flow_prob = prepare_and_predict(all_features)
            t_end = time.time()

            if flow_prob >= BEST_THRESHOLD:
                print(
                    f"[{time.strftime('%H:%M:%S')}] 🟢 心流状态 (Prob: {flow_prob:.2f}) | 耗时: {t_end - t_start:.2f}s")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] 🔴 认知枯竭 (Prob: {flow_prob:.2f}) -> 触发干预报警！")
                # 🔥 发送报警至后端
                try:
                    requests.post("http://127.0.0.1:8000/api/alert_depletion")
                except:
                    pass

            # 滑动窗口
            eeg_buf = eeg_buf[eeg_step:]
            physio_buf = physio_buf[physio_step:]
            et_buf = et_buf[et_step:]

        time.sleep(0.01)


if __name__ == '__main__':
    start_online_inference()