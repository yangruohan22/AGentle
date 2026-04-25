import sys
import time
import numpy as np
import mne
import matplotlib
import matplotlib.pyplot as plt
from pylsl import resolve_byprop, StreamInlet
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.use('Agg')  # 关键：无头模式，不在屏幕弹窗，直接存图片


def record_and_process(sub_id, duration):
    print(f"🎬 开始为 {sub_id} 录制 {duration} 秒基线数据...")

    # ✅ 1. 物理删除旧图，确保前端轮询时不会读到残骸
    os.makedirs('static_plots', exist_ok=True)
    stale_img = os.path.join('static_plots', 'current_ica.png')
    if os.path.exists(stale_img):
        os.remove(stale_img)

    # ✅ 2. 寻找三路 LSL 流
    print("📡 正在寻找 LSL 数据流 (脑电、生理、眼动)...")
    streams_eeg = resolve_byprop('name', 'Neuracle_EEG', timeout=5.0)
    streams_physio = resolve_byprop('name', 'Physio_NI6009', timeout=5.0)
    streams_et = resolve_byprop('name', 'EyeTracker', timeout=5.0)

    if not streams_eeg:
        print("❌ 错误：在局域网内未发现 'Neuracle_EEG' 信号！")
        sys.exit(1)
    if not streams_physio:
        print("❌ 错误：在局域网内未发现 'Physio_NI6009' 信号！")
        sys.exit(1)
    if not streams_et:
        print("❌ 错误：在局域网内未发现 'EyeTracker' 信号！")
        sys.exit(1)

    inlet_eeg = StreamInlet(streams_eeg[0])
    inlet_physio = StreamInlet(streams_physio[0])
    inlet_et = StreamInlet(streams_et[0])

    eeg_sfreq = int(inlet_eeg.info().nominal_srate()) or 1000
    physio_sfreq = int(inlet_physio.info().nominal_srate()) or 1000
    et_sfreq = int(inlet_et.info().nominal_srate()) or 1200

    target_eeg_samples = duration * eeg_sfreq
    target_physio_samples = duration * physio_sfreq
    target_et_samples = duration * et_sfreq

    eeg_buffer = []
    physio_buffer = []
    et_buffer = []

    print(f"⏳ 正在采集，请保持安静 {duration} 秒...")
    start_time = time.time()

    # ✅ 3. 并发收集三路数据
    while len(eeg_buffer) < target_eeg_samples or len(physio_buffer) < target_physio_samples or len(
            et_buffer) < target_et_samples:
        e_chunk, _ = inlet_eeg.pull_chunk(timeout=0.0)
        p_chunk, _ = inlet_physio.pull_chunk(timeout=0.0)
        et_chunk, _ = inlet_et.pull_chunk(timeout=0.0)

        if e_chunk: eeg_buffer.extend(e_chunk)
        if p_chunk: physio_buffer.extend(p_chunk)
        if et_chunk: et_buffer.extend(et_chunk)

        # 兜底防死循环
        if time.time() - start_time > duration + 10:
            print("⚠️ 采集超时，强行截断！")
            break
        time.sleep(0.005)

    print("✅ 数据收集完毕，正在保存原始数据...")

    # 将列表转换为 Numpy 数组并截取目标长度
    eeg_array = np.array(eeg_buffer[:target_eeg_samples])  # 形状 [N, channels]
    bio_array = np.array(physio_buffer[:target_physio_samples])
    et_array = np.array(et_buffer[:target_et_samples])

    config_dir = os.path.join("experiment_data", sub_id, "config")
    os.makedirs(config_dir, exist_ok=True)

    # ✅ 4. 核心修复：保存三路 .npy 数据，供计算特征均值使用！
    np.save(os.path.join(config_dir, f"{sub_id}_baseline_eeg.npy"), eeg_array)
    np.save(os.path.join(config_dir, f"{sub_id}_baseline_bio.npy"), bio_array)
    np.save(os.path.join(config_dir, f"{sub_id}_baseline_et.npy"), et_array)
    print("💾 原始 .npy 数据安全落盘！")

    print("✅ 开始进行 ICA 分解...")

    # 脑电转置并只取前 8 个通道 [8, samples]
    data_matrix = eeg_array.T[:8, :]

    # 5. MNE 预处理
    ch_names = ['AF3', 'AF4', 'F3', 'F1', 'Fz', 'F2', 'F4', 'Pz']
    info = mne.create_info(ch_names=ch_names, sfreq=eeg_sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data_matrix, info, verbose=False)
    raw.set_montage('standard_1020', on_missing='ignore')

    raw.notch_filter(freqs=50.0, fir_design='firwin', verbose=False)
    raw.filter(l_freq=1.0, h_freq=45.0, fir_design='firwin', verbose=False)

    # 6. 拟合 ICA 并保存模型（供实时阶段使用）
    ica = mne.preprocessing.ICA(n_components=8, method='picard', fit_params=dict(extended=True), random_state=97)
    ica.fit(raw, decim=10, verbose=False)
    fif_path = os.path.join(config_dir, f"{sub_id}_baseline_ica.fif")
    ica.save(fif_path, overwrite=True)

    # 7. 画出 8x3 诊断面板图
    sources = ica.get_sources(raw)
    ic_data = sources.get_data()
    times = raw.times

    mid_idx = len(times) // 2
    offset = int(10 * eeg_sfreq)
    start_idx = max(0, mid_idx - offset)
    end_idx = min(len(times), mid_idx + offset)
    time_seg = times[start_idx:end_idx]

    fig, axes = plt.subplots(nrows=8, ncols=3, figsize=(18, 30))
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    fig.suptitle(f'{sub_id} ICA 基线成分诊断面板', fontsize=26, y=0.92, fontweight='bold')

    for i in range(8):
        ax_topo = axes[i, 0]
        ica.plot_components(picks=[i], axes=[ax_topo], show=False, colorbar=False)
        ax_topo.set_title(f'成分 {i} 地形图', fontsize=14)

        ax_psd = axes[i, 1]
        psds, freqs = mne.time_frequency.psd_array_welch(ic_data[i], sfreq=eeg_sfreq, fmin=1, fmax=45, n_fft=2048,
                                                         verbose=False)
        alpha_mask, high_mask = (freqs >= 8) & (freqs <= 13), (freqs >= 25) & (freqs <= 45)
        pwr_ratio = np.mean(psds[high_mask]) / (np.mean(psds[alpha_mask]) + 1e-9)

        ax_psd.plot(freqs, 10 * np.log10(psds + 1e-9), color='k')
        ax_psd.set_title(f'成分 {i} 功率谱 (高低频比: {pwr_ratio:.2f})', fontsize=12)
        ax_psd.axvspan(8, 13, color='red', alpha=0.15)
        ax_psd.axvspan(25, 45, color='gray', alpha=0.15)

        ax_wave = axes[i, 2]
        ax_wave.plot(time_seg, ic_data[i, start_idx:end_idx], color='k', linewidth=0.8)
        ax_wave.set_title(f'成分 {i} 时域波形 (中间 20s)', fontsize=14)

    # 保存到静态目录供前端实时显示
    save_path = os.path.join('static_plots', 'current_ica.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=150)

    # 保存被试专属备份
    sub_img_path = os.path.join(config_dir, f"{sub_id}_ica_panel.png")
    fig.savefig(sub_img_path, bbox_inches='tight', dpi=150)

    plt.close(fig)
    print(f"🎉 任务准备就绪。专属图存至: {sub_img_path}")


if __name__ == "__main__":
    subject_id = sys.argv[1] if len(sys.argv) > 1 else "Unknown"
    # ✅ 读取前端传来的时长参数，默认 180 秒
    duration_sec = int(sys.argv[2]) if len(sys.argv) > 2 else 180
    record_and_process(subject_id, duration_sec)