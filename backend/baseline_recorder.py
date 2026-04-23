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
def record_and_process(sub_id):
    print(f"🎬 开始为 {sub_id} 录制 3 分钟基线数据...")

    # 1. 寻找 LSL 流
    # --- 修改第 20 行附近 ---
    print(f"🎬 开始为 {sub_id} 寻找 LSL 数据流...")

    # ✅ 修改：增加 timeout=5.0 参数。如果 5 秒找不到，它会返回空列表
    streams = resolve_byprop('name', 'Neuracle_EEG', timeout=5.0)

    if not streams:
        print("❌ 错误：在局域网内未发现 'Neuracle_EEG' 信号！请检查设备或模拟器是否开启。")
        # 这里可以尝试删除旧图，防止前端误读
        old_img = os.path.join(os.getcwd(), 'static_plots', 'current_ica.png')
        if os.path.exists(old_img): os.remove(old_img)
        sys.exit(1)  # 异常退出，这样 main.py 就能捕获到错误

    inlet = StreamInlet(streams[0])
    sfreq = int(inlet.info().nominal_srate())
    inlet = StreamInlet(streams[0])
    # 2. 收集 3 分钟 (180秒) 数据
    duration = 10
    target_samples = duration * sfreq
    eeg_buffer = []

    start_time = time.time()
    while len(eeg_buffer) < target_samples:
        chunk, _ = inlet.pull_chunk(timeout=1.0)
        if chunk:
            eeg_buffer.extend(chunk)
        # 兜底防死循环
        if time.time() - start_time > duration + 10:
            break

    print("✅ 数据收集完毕，开始进行 ICA 分解...")

    # 转置并只取前 8 个通道 [8, samples]
    data_matrix = np.array(eeg_buffer[:target_samples]).T[:8, :]

    # 3. MNE 预处理
    ch_names = ['AF3', 'AF4', 'F3', 'F1', 'Fz', 'F2', 'F4', 'Pz']
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data_matrix, info, verbose=False)
    raw.set_montage('standard_1020', on_missing='ignore')

    raw.notch_filter(freqs=50.0, fir_design='firwin', verbose=False)
    raw.filter(l_freq=1.0, h_freq=45.0, fir_design='firwin', verbose=False)

    # 4. 拟合 ICA 并保存模型（供实时阶段使用）
    ica = mne.preprocessing.ICA(n_components=8, method='picard', fit_params=dict(extended=True), random_state=97)
    ica.fit(raw, decim=10, verbose=False)
    config_dir = os.path.join("experiment_data", sub_id, "config")
    os.makedirs(config_dir, exist_ok=True)
    fif_path = os.path.join(config_dir, f"{sub_id}_baseline_ica.fif")
    ica.save(fif_path, overwrite=True)

    # 5. 画出 8x3 诊断面板图
    sources = ica.get_sources(raw)
    ic_data = sources.get_data()
    times = raw.times

    mid_idx = len(times) // 2
    offset = int(10 * sfreq)
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
        psds, freqs = mne.time_frequency.psd_array_welch(ic_data[i], sfreq=sfreq, fmin=1, fmax=45, n_fft=2048,
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

    save_path = os.path.join(os.getcwd(), 'static_plots', 'current_ica.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=150)

    # 保存图片给前端读取
    plt.close(fig)
    print(f"🎉 基线处理完成，ICA图已生成在: {save_path}")


if __name__ == "__main__":
    subject_id = sys.argv[1] if len(sys.argv) > 1 else "Unknown"
    record_and_process(subject_id)