import time
import math
import tobii_research as tr
from pylsl import StreamInfo, StreamOutlet, local_clock

# ================= 配置区 =================
STREAM_NAME = 'EyeTracker'
STREAM_TYPE = 'Gaze'
NUM_CHANNELS = 5

# 🌟 新增：看门狗状态记录器
watchdog = {
    'last_data_time': time.time(),
    'packet_count': 0
}


# ==========================================

def start_eyetracker_bridge():
    print("🔄 正在搜索局域网/USB内的 Tobii 眼动仪...")

    # 🌟 加入重试机制，防止服务假死
    eyetrackers = tr.find_all_eyetrackers()
    retries = 0
    while not eyetrackers and retries < 3:
        print(f"⏳ 未找到眼动仪，正在重试 ({retries + 1}/3)...")
        time.sleep(2)
        eyetrackers = tr.find_all_eyetrackers()
        retries += 1

    if not eyetrackers:
        print("❌ 彻底失败: 未发现眼动仪！请检查线缆，或在任务管理器重启 Tobii Service。")
        return

    et = eyetrackers[0]
    print(f"✅ 成功连接眼动仪: {et.model} (SN: {et.serial_number})")

    info = StreamInfo(STREAM_NAME, STREAM_TYPE, NUM_CHANNELS, 0, 'float32', f'tobii_{et.serial_number}')
    chns = info.desc().append_child("channels")
    for ch in ["Device_Timestamp", "Gaze_X", "Gaze_Y", "Pupil_L", "Pupil_R"]:
        chns.append_child("channel").append_child_value("label", ch)

    outlet = StreamOutlet(info)
    print(f"📡 LSL 数据流 [{STREAM_NAME}] 已建立！")

    def gaze_data_callback(gaze_data):
        try:
            # 🌟 每次收到数据，立刻“喂狗”并计数
            watchdog['last_data_time'] = time.time()
            watchdog['packet_count'] += 1

            l_g = gaze_data['left_gaze_point_on_display_area']
            r_g = gaze_data['right_gaze_point_on_display_area']
            l_p = gaze_data['left_pupil_diameter']
            r_p = gaze_data['right_pupil_diameter']
            sys_ts = gaze_data['system_time_stamp'] / 1e6

            xs = [v for v in [l_g[0], r_g[0]] if not math.isnan(v)]
            ys = [v for v in [l_g[1], r_g[1]] if not math.isnan(v)]

            avg_x = sum(xs) / len(xs) if xs else float('nan')
            avg_y = sum(ys) / len(ys) if ys else float('nan')

            payload = [sys_ts, avg_x, avg_y, l_p, r_p]
            outlet.push_sample(payload, local_clock())
        except Exception:
            pass

    print("🚀 正在实时拦截眼动数据并推流，按 Ctrl+C 停止...")
    print("-" * 50)
    et.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)

    try:
        # 🌟 升级主线程：看门狗巡逻
        while True:
            time_since_last_packet = time.time() - watchdog['last_data_time']

            # 如果超过 2.5 秒没收到数据，判定为断连
            if time_since_last_packet > 2.5:
                print(f"\n\n🚨 [看门狗报警] 超过 2.5 秒未收到眼动数据！")
                print("❌ Tobii 可能已进入休眠，或追踪丢失。桥接器将自动退出。")
                break

            print(f"\r[状态正常] 👁️ Tobii 采集中... 已推送包数: {watchdog['packet_count']}   ", end="", flush=True)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\n🛑 收到手动停止指令 (Ctrl+C)。")
    finally:
        print("🔌 正在取消订阅并释放设备...")
        et.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
        print("👋 眼动桥接器已安全退出。")


if __name__ == '__main__':
    start_eyetracker_bridge()