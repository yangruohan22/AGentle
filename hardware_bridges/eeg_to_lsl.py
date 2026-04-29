import os
import sys
import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet

# =========================================================================
# 🌟 终极防屏蔽导入法：强制 Python 优先读取当前目录下的 neuracle_lib
# =========================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from neuracle_lib.dataServer import DataServerThread
except ImportError as e:
    print(f"\n🚨 导入错误: {e}")
    sys.exit(1)

# ================= 🌟 核心配置区 🌟 =================
DEVICE_IP = '127.0.0.1'  # 博睿康主机 IP
PORT = 8712  # 默认端口
NUM_CHANNELS = 8  # 脑电通道数
SAMPLING_RATE = 1000  # 采样率
BUFFER_TIME = 3  # 3 秒缓存池


# ====================================================

def start_neuracle_bridge():
    print(f"\n🔄 正在初始化博睿康设备环境 (Neuracle, {NUM_CHANNELS}通道, {SAMPLING_RATE}Hz)...")

    # 1. 初始化官方 DataServer 线程
    thread_data_server = DataServerThread(
        device='Neuracle',
        n_chan=NUM_CHANNELS,
        srate=SAMPLING_RATE,
        t_buffer=BUFFER_TIME
    )

    print(f"🔌 正在尝试连接至 {DEVICE_IP}:{PORT} ...")
    notconnect = thread_data_server.connect(hostname=DEVICE_IP, port=PORT)

    if notconnect:
        print("❌ 致命错误: 无法连接到脑电设备！请检查 Recorder 是否开启了 DataServer 转发。")
        return
    else:
        thread_data_server.Daemon = True
        thread_data_server.start()
        print("✅ 成功连接博睿康数据引擎！")

    # 强制等待，给子线程留出数据填充时间
    print("⏳ 等待硬件缓存初始化...")
    time.sleep(1.5)

    # 2. 注册 LSL 广播出口
    info = StreamInfo('Neuracle_EEG', 'EEG', NUM_CHANNELS, SAMPLING_RATE, 'float32', 'neuracle_bci_001')
    outlet = StreamOutlet(info)
    print(f"📡 LSL [Neuracle_EEG] 局域网广播已开启！")
    print("🚀 正在实时推流 (抗积压模式)，按 Ctrl+C 停止...\n")

    last_data_time = time.time()
    WATCHDOG_THRESHOLD = 10.0

    try:
        while True:
            # 获取新到的点数
            nUpdate = thread_data_server.GetDataLenCount()

            # 🌟 [策略 A]：检测到严重积压 (超过3秒数据未读)，强制追赶清空
            if nUpdate > 3000:
                print(f"\n⚠️ 警告：检测到数据积压 ({nUpdate} 点)，执行强制同步...")
                data = thread_data_server.GetBufferData()
                thread_data_server.ResetDataLenCount()
                if data is not None and data.shape[1] > 0:
                    # 丢弃陈旧数据，只取最新的 1000 个采样点发送以保持实时性
                    new_data_chunk = data[:, -1000:]
                    outlet.push_chunk(new_data_chunk.T.tolist())
                    last_data_time = time.time()
                continue

            # 🌟 [策略 B]：正常推流逻辑
            if nUpdate > 0:
                data = thread_data_server.GetBufferData()
                if data is not None and data.shape[1] >= nUpdate:
                    last_data_time = time.time()

                    # 截取最新数据
                    new_data_chunk = data[:, -nUpdate:]

                    # 必须在 push 前 Reset，防止计数器在计算过程中继续累加导致溢出
                    thread_data_server.ResetDataLenCount()
                    outlet.push_chunk(new_data_chunk.T.tolist())

                    # 打印推流提示
                    if nUpdate > 50:  # 只有数据量够大时才打点
                        print(".", end="", flush=True)
            else:
                # 检查看门狗
                elapsed = time.time() - last_data_time
                if elapsed > WATCHDOG_THRESHOLD:
                    print(f"\n\n🚨 [看门狗报警] 持续 {WATCHDOG_THRESHOLD} 秒未收到新数据流。")
                    print("可能原因：1.Recorder停止采集 2.硬件断连 3.TCP链路阻塞")
                    break

            # 🌟 缩短轮询间隔至 10ms，提高在高频采样下的读取响应
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n🛑 收到手动停止指令 (Ctrl+C)。")
    except Exception as e:
        print(f"\n⚠️ 运行时异常: {e}")
    finally:
        print("🔌 正在安全释放设备连接...")
        thread_data_server.stop()
        print("👋 桥接器已安全退出。")


if __name__ == '__main__':
    start_neuracle_bridge()