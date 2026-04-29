import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
from pylsl import StreamInfo, StreamOutlet
import numpy as np
import time

# ================= 配置区 =================
DEVICE_NAME = "Dev1"  # NI MAX 里的设备名
SAMPLE_RATE = 1000  # 采样率 1000 Hz
CHUNK_SIZE = 100  # 每 0.1 秒读取一次 (100个点)
STREAM_NAME = "Physio_NI6009"
STREAM_TYPE = "Physiology"


# ==========================================

def start_streaming():
    # 1. 初始化 LSL 广播出口
    # 参数: 名字, 类型, 通道数(2个), 采样率, 数据类型, 唯一ID
    info = StreamInfo(STREAM_NAME, STREAM_TYPE, 2, SAMPLE_RATE, 'float32', 'ni6009_uid_999')

    # 给通道打上标签，方便后续接收端识别
    chns = info.desc().append_child("channels")
    for ch in ["GSR", "ECG"]:
        chns.append_child("channel").append_child_value("label", ch)

    outlet = StreamOutlet(info)
    print(f"📡 LSL 数据流 [{STREAM_NAME}] 已建立，等待接收端连接...")

    # 2. 初始化 NI 6009 硬件
    try:
        with nidaqmx.Task() as task:
            # 添加皮电 (CH10) -> AI 0
            task.ai_channels.add_ai_voltage_chan(
                f"{DEVICE_NAME}/ai0",
                terminal_config=TerminalConfiguration.RSE
            )
            # 添加心电 (CH14) -> AI 1
            task.ai_channels.add_ai_voltage_chan(
                f"{DEVICE_NAME}/ai1",
                terminal_config=TerminalConfiguration.RSE
            )

            # 配置硬件时钟 (连续模式)
            task.timing.cfg_samp_clk_timing(
                rate=SAMPLE_RATE,
                sample_mode=AcquisitionType.CONTINUOUS
            )

            print(f"✅ 成功连接 NI 6009 ({DEVICE_NAME})，开始采集并推送数据...")
            print("-" * 50)

            task.start()

            while True:
                # 读取数据，data 的结构是: [[皮电的100个点], [心电的100个点]]
                data = task.read(number_of_samples_per_channel=CHUNK_SIZE)

                # 3. 转换数据格式以适应 LSL
                # LSL 需要的格式是: [[时刻1皮电, 时刻1心电], [时刻2皮电, 时刻2心电], ...]
                # 我们使用 NumPy 的转置矩阵 (T) 功能来一键完成转换
                chunk_to_push = np.array(data).T.tolist()

                # 推送到 LSL 网络
                outlet.push_chunk(chunk_to_push)

                # 4. 控制台实时验证 (提取均值和最新值)
                gsr_array = np.array(data[0])
                ecg_array = np.array(data[1])

                # 皮电看平稳均值，心电看剧烈的瞬时波动
                gsr_mean = np.mean(gsr_array)
                ecg_latest = ecg_array[-1]

                # 使用回车符 \r 让控制台单行刷新，不刷屏
                print(
                    f"\r[状态正常] 🟢 GSR (皮电) 均值: {gsr_mean: .4f} V  |  ❤️ ECG (心电) 瞬时: {ecg_latest: .4f} V   ",
                    end="", flush=True)

    except nidaqmx.errors.DaqError as e:
        print(f"\n\n❌ 硬件采集错误：\n{e}")
    except KeyboardInterrupt:
        print("\n\n⏹️ 数据流广播已手动停止。")


if __name__ == "__main__":
    start_streaming()