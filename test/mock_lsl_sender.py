import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet


def start_mock_system():
    print("🚀 A-Gentle 虚拟硬件环境启动中 (Chunk 优化版)...")

    # 1. 模拟脑电 (8通道, 1000Hz)
    info_eeg = StreamInfo('Neuracle_EEG', 'EEG', 8, 1000, 'float32', 'mock_eeg_001')
    outlet_eeg = StreamOutlet(info_eeg)

    # 2. 模拟生理仪 (2通道: 0-GSR, 1-ECG, 1000Hz)
    info_physio = StreamInfo('Physio_NI6009', 'Physio', 2, 1000, 'float32', 'mock_physio_001')
    outlet_physio = StreamOutlet(info_physio)

    # 3. 模拟眼动仪 (5通道, 1200Hz)
    info_et = StreamInfo('EyeTracker', 'ET', 5, 1200, 'float32', 'mock_et_001')
    outlet_et = StreamOutlet(info_et)

    print("✅ 所有虚拟 LSL 流已就绪。正在以 10Hz 频率批量压入信号...")

    # 我们设定每 0.1 秒发一次货 (10Hz 的循环频率，Python 绝对能稳稳 hold 住)
    # 脑电和生理 1000Hz，0.1秒就是 100 个点
    # 眼动 1200Hz，0.1秒就是 120 个点
    chunk_eeg_size = 100
    chunk_et_size = 120

    count = 0
    while True:
        cycle_start = time.time()

        # --- 批量模拟脑电数据 (100个点) ---
        eeg_chunk = np.random.randn(chunk_eeg_size, 8) * 50.0
        outlet_eeg.push_chunk(eeg_chunk.tolist())

        # --- 批量模拟生理数据 (100个点) ---
        gsr_chunk = 5.0 + np.sin((count * 100 + np.arange(chunk_eeg_size)) * 0.01) * 0.5 + np.random.randn(
            chunk_eeg_size) * 0.01
        ecg_chunk = np.random.randn(chunk_eeg_size) * 0.1
        physio_chunk = np.column_stack((gsr_chunk, ecg_chunk))
        outlet_physio.push_chunk(physio_chunk.tolist())

        # --- 批量模拟眼动数据 (120个点) ---
        gx_chunk = 0.5 + np.random.randn(chunk_et_size) * 0.05
        gy_chunk = 0.5 + np.random.randn(chunk_et_size) * 0.05
        pupil_chunk = 3.0 + np.random.randn(chunk_et_size) * 0.1
        zeros_chunk = np.zeros(chunk_et_size)
        et_chunk = np.column_stack((zeros_chunk, gx_chunk, gy_chunk, pupil_chunk, pupil_chunk))
        outlet_et.push_chunk(et_chunk.tolist())

        count += 1

        # 动态休眠，确保每次循环极其精准地耗时 0.1 秒
        elapsed = time.time() - cycle_start
        sleep_time = max(0, 0.1 - elapsed)
        time.sleep(sleep_time)


if __name__ == "__main__":
    try:
        start_mock_system()
    except KeyboardInterrupt:
        print("\n🛑 虚拟环境已关闭。")