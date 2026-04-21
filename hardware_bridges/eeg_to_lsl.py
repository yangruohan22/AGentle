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
    # 使用 insert(0, ...) 强行把 hardware_bridges 塞到系统搜索路径的最前面
    # 这样就算你根目录有同名文件夹，Python 也会优先用这里的！
    sys.path.insert(0, CURRENT_DIR)

try:
    from neuracle_lib.dataServer import DataServerThread
except ImportError as e:
    print(f"\n🚨 极其异常的导入错误！\n错误详情: {e}")
    print(f"当前 Python 搜索的绝对路径是: {CURRENT_DIR}\\neuracle_lib")
    print("请检查该文件夹下是否生成了 __pycache__ 文件夹，如果有，请直接删掉它然后重试！")
    sys.exit(1)

# ================= 🌟 核心配置区 🌟 =================
DEVICE_IP = '127.0.0.1'  # 你的博睿康主机 IP (本机测试填 127.0.0.1)
PORT = 8712  # 默认端口
NUM_CHANNELS = 65  # 脑电帽真实通道数 (如需修改请按实际填)
SAMPLING_RATE = 1000  # 采样率
BUFFER_TIME = 3  # 官方默认的 3 秒缓存池


# ====================================================

def start_neuracle_bridge():
    print(f"\n🔄 正在初始化博睿康设备环境 (设备名: Neuracle, {NUM_CHANNELS}通道, {SAMPLING_RATE}Hz)...")

    # 完全使用官方库初始化
    thread_data_server = DataServerThread(
        device='Neuracle',
        n_chan=NUM_CHANNELS,
        srate=SAMPLING_RATE,
        t_buffer=BUFFER_TIME
    )

    print(f"🔌 正在尝试连接至 {DEVICE_IP}:{PORT} ...")
    notconnect = thread_data_server.connect(hostname=DEVICE_IP, port=PORT)

    if notconnect:
        print("❌ 致命错误: 无法连接到脑电设备！请检查 IP 地址和网络广播设置。")
        return
    else:
        thread_data_server.Daemon = True
        thread_data_server.start()
        print("✅ 成功连接博睿康数据引擎！")

    # 注册 LSL 广播出口
    info = StreamInfo('Neuracle_EEG', 'EEG', NUM_CHANNELS, SAMPLING_RATE, 'float32', 'neuracle_bci_001')
    outlet = StreamOutlet(info)
    print(f"📡 LSL [Neuracle_EEG] 局域网广播已开启！\n")
    print("🚀 正在实时拦截数据并推流，按 Ctrl+C 停止...")

    try:
        while True:
            # 使用官方接口获取新数据长度
            nUpdate = thread_data_server.GetDataLenCount()

            if nUpdate > 0:
                # 使用官方接口获取缓存池数据
                data = thread_data_server.GetBufferData()
                thread_data_server.ResetDataLenCount()

                # 🔪 只截取矩阵最末尾的 nUpdate 个最新数据点
                new_data_chunk = data[:, -nUpdate:]

                # 转换格式并推入 LSL
                chunk_to_push = new_data_chunk.T.tolist()
                outlet.push_chunk(chunk_to_push)

            # 20ms 轮询，保证极低延迟
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n🛑 收到手动停止指令 (Ctrl+C)。")
    except Exception as e:
        print(f"\n⚠️ 运行时发生异常: {e}")
    finally:
        print("🔌 正在安全关闭底层 TCP 线程...")
        thread_data_server.stop()
        print("👋 桥接器已退出。")


if __name__ == '__main__':
    start_neuracle_bridge()