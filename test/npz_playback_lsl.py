import time
import numpy as np
import os
import glob
from pylsl import StreamInfo, StreamOutlet

# ================= 1. 配置区 =================
# 📁 你的数据所在的总文件夹路径
DATA_DIR = r'D:\THU\fourthfall\research\graduation_design\dataAnalysis\processed_samples'

# 🎯 你想拼合放映的目标前缀 (脚本会自动寻找该前缀下所有的 _M1, _M2... 文件)
TARGET_PREFIX = 'Sub10_L1'

EEG_SFREQ = 1000
PHYSIO_SFREQ = 1000
ET_SFREQ = 1200

# 提取你代码里的通道索引
GSR_IDX, ECG_IDX = 2, 3


def start_playback_system():
    print(f"📦 正在扫描目录，寻找前缀为 {TARGET_PREFIX} 的所有切片...")

    # 自动搜索匹配的文件
    search_pattern = os.path.join(DATA_DIR, f"{TARGET_PREFIX}_M*.npz")
    file_list = glob.glob(search_pattern)

    if not file_list:
        print(f"❌ 找不到任何匹配 {TARGET_PREFIX}_M*.npz 的文件，请检查路径和前缀！")
        return

    # 🌟 关键：按照 M 后面的数字大小进行严格的顺序排序 (确保 M2 在 M10 前面)
    try:
        file_list.sort(key=lambda x: int(os.path.basename(x).split('_M')[-1].replace('.npz', '')))
    except Exception as e:
        print(f"⚠️ 文件名解析排序失败，将使用默认排序: {e}")
        file_list.sort()

    print(f"🔗 发现 {len(file_list)} 个切片，准备进行无缝拼接:")
    for f in file_list:
        print(f"   - {os.path.basename(f)}")

    eeg_list, physio_list, et_list = [], [], []

    # 依次加载并放入列表
    for f_path in file_list:
        loader = np.load(f_path)
        eeg_list.append(loader['eeg'].T)  # EEG 转置
        physio_list.append(loader['bio'][:, [GSR_IDX, ECG_IDX]])  # 提取所需的生理通道
        et_list.append(loader['et'])

    # 🌟 将所有列表垂直拼接成连续的长矩阵！
    eeg_data = np.vstack(eeg_list)
    physio_data = np.vstack(physio_list)
    et_data = np.vstack(et_list)

    total_eeg_samples = eeg_data.shape[0]
    total_et_samples = et_data.shape[0]
    total_minutes = total_eeg_samples / (EEG_SFREQ * 60)
    print(f"✅ 无缝拼接成功！获得连续长卷：总时长约 {total_minutes:.2f} 分钟。")

    # ================= 2. 建立 LSL 虚拟出水口 =================
    print("🚀 正在注册虚拟 LSL 广播节点...")
    info_eeg = StreamInfo('Neuracle_EEG', 'EEG', 8, EEG_SFREQ, 'float32', 'playback_eeg')
    outlet_eeg = StreamOutlet(info_eeg)

    info_physio = StreamInfo('Physio_NI6009', 'Physio', 2, PHYSIO_SFREQ, 'float32', 'playback_physio')
    outlet_physio = StreamOutlet(info_physio)

    info_et = StreamInfo('EyeTracker', 'ET', 5, ET_SFREQ, 'float32', 'playback_et')
    outlet_et = StreamOutlet(info_et)

    print("🟢 LSL 广播已就绪！开始以 1 倍速实时长卷回放...")

    # ================= 3. 高精度 Chunk 回放循环 =================
    # 设定每 0.1 秒发一次货
    chunk_duration = 0.1
    eeg_chunk_size = int(EEG_SFREQ * chunk_duration)  # 100
    physio_chunk_size = int(PHYSIO_SFREQ * chunk_duration)  # 100
    et_chunk_size = int(ET_SFREQ * chunk_duration)  # 120

    eeg_cursor = 0
    et_cursor = 0

    while True:
        cycle_start = time.time()

        # 如果这几个拼接起来的长文件全都放完了，才自动回到开头循环播放！
        if eeg_cursor + eeg_chunk_size > total_eeg_samples:
            eeg_cursor = 0
            et_cursor = 0
            print("\n🔁 整个长卷已播放到底，正在重新从头循环播放...")

        # 切片截取当前 Chunk
        eeg_chunk = eeg_data[eeg_cursor: eeg_cursor + eeg_chunk_size, :]
        physio_chunk = physio_data[eeg_cursor: eeg_cursor + physio_chunk_size, :]
        et_chunk = et_data[et_cursor: et_cursor + et_chunk_size, :]

        # 推入 LSL 水管
        outlet_eeg.push_chunk(eeg_chunk.tolist())
        outlet_physio.push_chunk(physio_chunk.tolist())
        outlet_et.push_chunk(et_chunk.tolist())

        # 移动游标
        eeg_cursor += eeg_chunk_size
        et_cursor += et_chunk_size

        # 严格的动态休眠，保证绝对的 1 倍速实时性
        elapsed = time.time() - cycle_start
        sleep_time = max(0.0, chunk_duration - elapsed)
        time.sleep(sleep_time)


if __name__ == "__main__":
    try:
        start_playback_system()
    except KeyboardInterrupt:
        print("\n🛑 回放已手动终止。")