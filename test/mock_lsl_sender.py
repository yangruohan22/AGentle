import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet

def start_mock_system():
    print("🚀 A-Gentle 虚拟硬件环境启动中...")

    # 1. 模拟脑电 (8通道, 1000Hz)
    info_eeg = StreamInfo('Neuracle_EEG', 'EEG', 8, 1000, 'float32', 'mock_eeg_001')
    outlet_eeg = StreamOutlet(info_eeg)

    # 2. 模拟生理仪 (2通道: 0-GSR, 1-ECG, 1000Hz)
    info_physio = StreamInfo('Physio_NI6009', 'Physio', 2, 1000, 'float32', 'mock_physio_001')
    outlet_physio = StreamOutlet(info_physio)

    # 3. 模拟眼动仪 (5通道: X, Y, Pupil_L, Pupil_R, 其他, 1200Hz)
    info_et = StreamInfo('EyeTracker', 'ET', 5, 1200, 'float32', 'mock_et_001')
    outlet_et = StreamOutlet(info_et)

    print("✅ 所有虚拟 LSL 流已就绪。正在模拟信号...")

    count = 0
    while True:
        # --- 模拟脑电数据 (微伏级随机信号) ---
        eeg_sample = np.random.randn(8) * 50.0
        outlet_eeg.push_sample(eeg_sample)

        # --- 模拟生理数据 (GSR 缓慢漂移，ECG 随机噪声) ---
        # GSR 模拟：基础电导 5.0 + 微小波动
        gsr = 5.0 + np.sin(count * 0.01) * 0.5 + np.random.randn() * 0.01
        # ECG 模拟：简单随机值 (注意：真实 nk 算 HRV 可能需要更真实的波形，测试时可先看流程)
        ecg = np.random.randn() * 0.1
        outlet_physio.push_sample([gsr, ecg])

        # --- 模拟眼动数据 (坐标 0~1) ---
        # 模拟注视点在屏幕中心小范围抖动
        gx = 0.5 + np.random.randn() * 0.05
        gy = 0.5 + np.random.randn() * 0.05
        pupil = 3.0 + np.random.randn() * 0.1
        outlet_et.push_sample([0, gx, gy, pupil, pupil])

        count += 1
        # 按照 1000Hz 的步长休眠 (1ms)
        time.sleep(0.001)

if __name__ == "__main__":
    try:
        start_mock_system()
    except KeyboardInterrupt:
        print("\n🛑 虚拟环境已关闭。")