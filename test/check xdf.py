import pyxdf

# 把这里的路径换成你那个变成乱码的文件的真实路径
file_path = "C:/Users/yxy/Documents/CurrentStudy/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf"

print(f"🔍 正在解析 XDF 文件: {file_path}")

try:
    # load_xdf 会返回所有的流(streams)和全局头文件(header)
    streams, header = pyxdf.load_xdf(file_path)

    print(f"\n✅ 成功读取！共发现 {len(streams)} 条数据流：\n" + "-" * 50)

    for i, stream in enumerate(streams):
        info = stream['info']
        name = info['name'][0]
        stype = info['type'][0]
        channel_count = int(info['channel_count'][0])
        srate = info['nominal_srate'][0]

        # 提取真实的数据矩阵长度
        time_series = stream['time_series']
        data_points = len(time_series)

        print(f"🌊 流 [{i + 1}]: {name} (类型: {stype})")
        print(f"   ├─ 通道数: {channel_count}")
        print(f"   ├─ 采样率: {srate} Hz")
        print(f"   └─ 包含数据点数: {data_points}")
        print("-" * 50)

except Exception as e:
    print(f"❌ 解析失败，报错信息：{e}")