import time
import joblib
import numpy as np
import pandas as pd
import mne
from mne_icalabel import label_components
import neurokit2 as nk
from scipy.stats import entropy
from scipy.ndimage import binary_closing
from pylsl import resolve_stream, StreamInlet
import warnings

# ================= 0. 环境补丁与配置 =================
warnings.filterwarnings('ignore')
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'trapz'): np.trapz = np.trapezoid

WINDOW_SIZE_SEC = 60.0  # 核心：保留 60 秒完整长窗口
STEP_SIZE_SEC = 5.0  # 步长：每隔 5 秒更新一次状态 (60秒数据滑动 5秒)
BEST_THRESHOLD = 0.42  # 总冠军触发阈值

EEG_SFREQ = 1000
PHYSIO_SFREQ = 1000
ET_SFREQ = 1200  # 假设眼动仪为 1200Hz

# ================= 1. 唤醒模型武器库 =================
print("🧠 正在唤醒 A-Gentle AI 引擎...")
live_model = joblib.load('agentle_lgbm_champion.pkl')
live_scaler = joblib.load('agentle_scaler.pkl')
expected_features = joblib.load('agentle_features.pkl')
print(f"✅ 模型加载完毕！严格要求对齐 {len(expected_features)} 个特征。")


# ================= 2. 实时预处理 (复刻 Stage A & B) =================
def preprocess_eeg_realtime(eeg_data_60s):
    """
    接收 60 秒的脑电数据，执行陷波、带通、以及 ICA 自动去噪。
    eeg_data_60s 形状: [8, 60000]
    """
    ch_names = ['AF3', 'AF4', 'F3', 'F1', 'Fz', 'F2', 'F4', 'Pz']
    info = mne.create_info(ch_names=ch_names, sfreq=EEG_SFREQ, ch_types='eeg')

    # 动态构建 MNE Raw 对象
    raw = mne.io.RawArray(eeg_data_60s, info, verbose=False)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')

    # 陷波与带通
    raw.notch_filter(freqs=50.0, fir_design='firwin', verbose=False)
    raw.filter(l_freq=1.0, h_freq=45.0, fir_design='firwin', verbose=False)

    # ICA 自动清洗
    try:
        ica = mne.preprocessing.ICA(n_components=8, method='picard', fit_params=dict(extended=True), random_state=97)
        ica.fit(raw, decim=10, verbose=False)
        ic_labels = label_components(raw, ica, method='iclabel')

        bad_components = [idx for idx, label in enumerate(ic_labels['labels'])
                          if label in ['eye', 'muscle', 'channel noise', 'line noise', 'heart']]
        ica.exclude = bad_components
        clean_raw = ica.apply(raw.copy(), verbose=False)

        return clean_raw.get_data()  # 返回清洗后的 [8, 60000] 矩阵
    except Exception as e:
        print(f"⚠️ 实时 ICA 清洗失败，退化为仅带通滤波: {e}")
        return raw.get_data()


# ================= 3. 核心特征提取 (原汁原味复刻) =================
# [为保持原样，以下四个函数直接复制你提供的原代码，无任何逻辑修改]

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
            raise ValueError(f"🚨 灾难拦截：缺少特征 '{feat_name}'！")

        # 处理可能出现的 NaN (用 0 填补，防止模型崩溃)
        val = features_dict[feat_name]
        aligned_vector.append(0.0 if np.isnan(val) else val)

    X_raw = np.array([aligned_vector])
    X_scaled = live_scaler.transform(X_raw)
    prob_state_1 = live_model.predict_proba(X_scaled)[0][1]
    return prob_state_1


# ================= 5. LSL 三路并发主循环 =================
def start_online_inference():
    print("\n📡 正在局域网内寻找数据流...")

    inlet_eeg = StreamInlet(resolve_stream('name', 'Neuracle_EEG')[0])
    inlet_physio = StreamInlet(resolve_stream('name', 'Physio_NI6009')[0])
    # 假设眼动仪的 LSL 流名称为 'EyeTracker'
    inlet_et = StreamInlet(resolve_stream('name', 'EyeTracker')[0])
    print(f"✅ 成功锁定所有 LSL 流！准备收集第一个 60 秒...")

    eeg_buf, physio_buf, et_buf = [], [], []
    eeg_win = int(WINDOW_SIZE_SEC * EEG_SFREQ)
    physio_win = int(WINDOW_SIZE_SEC * PHYSIO_SFREQ)
    et_win = int(WINDOW_SIZE_SEC * ET_SFREQ)

    # 步长转化为样本数
    eeg_step = int(STEP_SIZE_SEC * EEG_SFREQ)
    physio_step = int(STEP_SIZE_SEC * PHYSIO_SFREQ)
    et_step = int(STEP_SIZE_SEC * ET_SFREQ)

    while True:
        e_chunk, _ = inlet_eeg.pull_chunk(timeout=0.0)
        p_chunk, _ = inlet_physio.pull_chunk(timeout=0.0)
        et_chunk, _ = inlet_et.pull_chunk(timeout=0.0)

        if e_chunk: eeg_buf.extend(e_chunk)
        if p_chunk: physio_buf.extend(p_chunk)
        if et_chunk: et_buf.extend(et_chunk)

        # 只要三个池子都攒够了 60 秒，立刻触发一次推断！
        if len(eeg_buf) >= eeg_win and len(physio_buf) >= physio_win and len(et_buf) >= et_win:

            # 1. 切片提取过去 60 秒的数据
            current_eeg_raw = np.array(eeg_buf[:eeg_win]).T  # 转为 [65, 60000]
            current_physio = np.array(physio_buf[:physio_win]).T  # 转为 [2, 60000]
            current_et = np.array(et_buf[:et_win])  # 眼动特征代码期望 [N, 通道数]

            # 💡 假设脑电只需要前 8 个通道
            current_eeg_raw = current_eeg_raw[:8, :]

            # 2. 实时预处理 (MNE 滤波与 ICA)
            cleaned_eeg = preprocess_eeg_realtime(current_eeg_raw)

            # 3. 三路并发提取特征，合成最终字典
            all_features = {}
            all_features.update(get_eeg_features_full_stream(cleaned_eeg))
            # 假设 GSR 在通道 0，ECG 在通道 1
            all_features.update(get_gsr_features_stream_optimized(current_physio[0]))
            all_features.update(get_ecg_features_robust_stream(current_physio[1]))
            all_features.update(get_et_features_enhanced(current_et))

            # 4. 终极模型推断
            flow_prob = prepare_and_predict(all_features)

            if flow_prob >= BEST_THRESHOLD:
                print(f"[{time.strftime('%H:%M:%S')}] 🟢 心流状态 (Prob: {flow_prob:.2f})")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] 🔴 认知枯竭 (Prob: {flow_prob:.2f}) -> 准备干预！")

            # 5. 滑动窗口：丢弃最老的 STEP_SIZE_SEC（5秒）数据，等待新血注入
            eeg_buf = eeg_buf[eeg_step:]
            physio_buf = physio_buf[physio_step:]
            et_buf = et_buf[et_step:]

        time.sleep(0.01)


if __name__ == '__main__':
    start_online_inference()