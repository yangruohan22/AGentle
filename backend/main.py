import asyncio
import json
import re
import subprocess
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
from datetime import datetime
import os

# ================= 新增：引入 LSL 库 =================
try:
    from pylsl import StreamInfo, StreamOutlet
except ImportError:
    StreamInfo, StreamOutlet = None, None
# =====================================================

# 获取当前 backend 文件夹的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
static_path = os.path.join(current_dir, "static_plots")
os.makedirs(static_path, exist_ok=True)

app = FastAPI()

# ================= 新增：初始化 LSL Marker 出口 =================
try:
    if StreamInfo and StreamOutlet:
        info = StreamInfo('AGentle_Marker', 'Markers', 1, 0, 'string', 'agentle_uid_12345')
        marker_outlet = StreamOutlet(info)
        print("✅ LSL Marker 广播通道初始化成功！")
    else:
        marker_outlet = None
        print("⚠️ 未检测到 pylsl 库，LSL Marker 打标暂不可用（如有需要请 pip install pylsl）。")
except Exception as e:
    print(f"⚠️ LSL Marker 初始化失败: {e}")
    marker_outlet = None
# ================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static_plots", exist_ok=True)
os.makedirs("experiment_data", exist_ok=True)

app.mount("/static", StaticFiles(directory=static_path), name="static")

# ==========================================
# 大模型 (LLM) 引擎配置
# ==========================================
KIMI_API_KEY = "sk-KnNJv6ikxTJV6R2VIXdN1SJ7LBZpzBEW0YcVkHqUTuvclgxB"
client = AsyncOpenAI(api_key=KIMI_API_KEY, base_url="https://api.moonshot.cn/v1")

active_websockets = []
active_inference_process = None
active_baseline_process = None

class InferenceRequest(BaseModel):
    sub_id: str

class SetupData(BaseModel):
    sub_id: str
    group: int
    duration: int = 180
    # task_id: int # 去除必填限制以兼容前端


@app.post("/api/start_baseline")
async def start_baseline(data: SetupData):
    global active_baseline_process
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 开始基线采集 (Sub: {data.sub_id})")
    try:
        # 如果还有上一个没死透的基线进程，先清理掉
        if active_baseline_process is not None:
            active_baseline_process.terminate()

        active_baseline_process = subprocess.Popen(["python", "baseline_recorder.py", data.sub_id, str(data.duration)])
    except Exception as e:
        print(f"启动录制脚本失败: {e}")
    return {"status": "success"}


# ================= 新增：前端用来轮询进程是否彻底结束的接口 =================
@app.get("/api/check_baseline_status")
async def check_baseline_status():
    global active_baseline_process
    if active_baseline_process is None:
        return {"status": "idle"}

    ret_code = active_baseline_process.poll()
    if ret_code is None:
        return {"status": "running"}  # 进程还在跑，画图还没彻底完成
    else:
        return {"status": "done"}  # 进程彻底结束，图片100%安全落盘！

class ICAExcludes(BaseModel):
    sub_id: str
    exclude_indices: str


@app.post("/api/submit_ica")
async def submit_ica(data: ICAExcludes):
    indices = [int(x.strip()) for x in data.exclude_indices.split(",") if x.strip()]

    # ✅ 修改：保存到被试专属文件夹
    config_dir = f"experiment_data/{data.sub_id}/config"
    os.makedirs(config_dir, exist_ok=True)
    config_path = f"{config_dir}/{data.sub_id}_ica_config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({"manual_excludes": indices}, f, ensure_ascii=False)
    return {"status": "success"}

class InferenceRequest(BaseModel):
    sub_id: str
# ================= 新增：一键启动推断引擎 =================
@app.post("/api/start_inference")
async def start_inference(data: InferenceRequest):
    global active_inference_process
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🧠 为被试 {data.sub_id} 启动心流探测引擎...")
    try:
        # 🚨 核心防呆：如果有旧的推断进程还在跑，强行终止它！
        if active_inference_process is not None:
            print("⚠️ 检测到上一个推断引擎仍在运行，正在强制终止清理水池...")
            active_inference_process.terminate()
            active_inference_process.wait()  # 确保彻底死透

        cwd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'online_system'))

        # 启动全新的进程，此时水池是空的（需要重新注水60秒），且会重新读取最新的 ICA JSON
        active_inference_process = subprocess.Popen(["python", "main_inference.py", data.sub_id], cwd=cwd_path)

        return {"status": "success"}
    except Exception as e:
        print(f"启动失败: {e}")
        return {"status": "error"}
# ================= 修改：分类保存实验数据 =================
@app.post("/api/save_experiment")
async def save_experiment(payload: dict):
    sub_id = payload.get("metadata", {}).get("sub_id", "UnknownSub")
    base_dir = f"experiment_data/{sub_id}"

    os.makedirs(f"{base_dir}/questionnaires", exist_ok=True)
    os.makedirs(f"{base_dir}/behavior", exist_ok=True)
    os.makedirs(f"{base_dir}/text_output", exist_ok=True)
    os.makedirs(f"{base_dir}/chat_logs", exist_ok=True)

    with open(f"{base_dir}/questionnaires/survey_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "pretest": payload.get("pretest"),
            "posttest": payload.get("posttest"),
            "tasks_midtest": [t.get("midtest") for t in payload.get("tasks", [])]
        }, f, ensure_ascii=False, indent=2)

    for idx, task in enumerate(payload.get("tasks", [])):
        with open(f"{base_dir}/text_output/task_{idx + 1}_story.txt", "w", encoding="utf-8") as f:
            f.write(task.get("final_text", ""))
        with open(f"{base_dir}/behavior/task_{idx + 1}_keystrokes.json", "w", encoding="utf-8") as f:
            json.dump(task.get("keystrokes", []), f, ensure_ascii=False, indent=2)
        with open(f"{base_dir}/chat_logs/task_{idx + 1}_chats.json", "w", encoding="utf-8") as f:
            json.dump(task.get("chat_log", []), f, ensure_ascii=False, indent=2)

    with open(f"{base_dir}/{sub_id}_UltimateLog.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"🎉 实验数据分类归档至: {base_dir}/")
    return {"status": "success"}


@app.post("/api/alert_depletion")
async def alert_depletion():
    # 纯转发：告诉所有前端，生理枯竭了！触发组 3 的逻辑
    for ws in active_websockets:
        await ws.send_json({"type": "physiological_alert"})
    return {"status": "success"}

# =========================================================================
# ========================= 新增：打标与单步覆盖 API 区 =========================
# =========================================================================

class MarkerData(BaseModel):
    event: str
    abs_time: str

@app.post("/api/send_marker")
async def send_marker(data: MarkerData):
    if marker_outlet:
        # 将事件和绝对时间戳拼成字符串发给 LSL
        marker_str = f"EVENT:{data.event}|ABS_TIME:{data.abs_time}"
        marker_outlet.push_sample([marker_str])
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚩 LSL打标: {data.event}")
    return {"status": "success"}

class PretestData(BaseModel):
    sub_id: str
    answers: list

@app.post("/api/save_pretest")
async def save_pretest(data: PretestData):
    dir_path = f"experiment_data/{data.sub_id}/questionnaires"
    os.makedirs(dir_path, exist_ok=True)
    with open(f"{dir_path}/pretest.json", "w", encoding="utf-8") as f:
        json.dump(data.answers, f, ensure_ascii=False, indent=2)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📝 前测问卷已即时保存/覆盖")
    return {"status": "success"}

class PosttestData(BaseModel):
    sub_id: str
    answers: dict

@app.post("/api/save_posttest")
async def save_posttest(data: PosttestData):
    dir_path = f"experiment_data/{data.sub_id}/questionnaires"
    os.makedirs(dir_path, exist_ok=True)
    with open(f"{dir_path}/posttest.json", "w", encoding="utf-8") as f:
        json.dump(data.answers, f, ensure_ascii=False, indent=2)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📝 后测问卷已即时保存/覆盖")
    return {"status": "success"}

class PartialTaskData(BaseModel):
    sub_id: str
    task_id: int
    text: str
    keystrokes: list
    chat_log: list
    survey_mid: dict

@app.post("/api/save_partial_task")
async def save_partial_task(data: PartialTaskData):
    base_dir = f"experiment_data/{data.sub_id}"
    os.makedirs(f"{base_dir}/text_output", exist_ok=True)
    os.makedirs(f"{base_dir}/behavior", exist_ok=True)
    os.makedirs(f"{base_dir}/chat_logs", exist_ok=True)
    os.makedirs(f"{base_dir}/questionnaires", exist_ok=True)

    # 用 "w" 模式覆盖写入，完美支持被试重新做某一个任务
    with open(f"{base_dir}/text_output/task_{data.task_id}_story.txt", "w", encoding="utf-8") as f:
        f.write(data.text)
    with open(f"{base_dir}/behavior/task_{data.task_id}_keystrokes.json", "w", encoding="utf-8") as f:
        json.dump(data.keystrokes, f, ensure_ascii=False, indent=2)
    with open(f"{base_dir}/chat_logs/task_{data.task_id}_chats.json", "w", encoding="utf-8") as f:
        json.dump(data.chat_log, f, ensure_ascii=False, indent=2)
    with open(f"{base_dir}/questionnaires/task_{data.task_id}_midtest.json", "w", encoding="utf-8") as f:
        json.dump(data.survey_mid, f, ensure_ascii=False, indent=2)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 💾 任务 {data.task_id} 的全部数据(含中测)已安全覆盖/备份")
    return {"status": "success"}


# ==========================================
# 核心 Agent 调用函数
# ==========================================
async def call_agent(role: str, sys_prompt: str, user_prompt: str):
    try:
        resp = await client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1
        )
        return {"role": role, "content": resp.choices[0].message.content.strip()}
    except Exception as e:
        print(f"Agent {role} 调用失败: {e}")
        return {"role": role, "content": "（由于系统干扰，该角色的信号暂时丢失）"}

async def call_env_agent(theme: str, current_text: str):
    """专门负责返回 JSON 调整环境颜色的主脑"""
    sys_prompt = """你是一个负责控制沉浸式写作环境的AI主脑。请根据作者正在写作的内容情绪和设定的世界观，返回一个JSON来改变整个网页的氛围。
    必须严格返回合法的JSON，格式如下：
    {
      "page_bg": "#十六进制颜色码",      // 控制整个网页的最底层大背景（建议暗色调、低饱和度）
      "box_bg": "#十六进制颜色码",       // 控制文本输入框的背景色（应与page_bg有所区分，但保持协调）
      "text_color": "#十六进制颜色码",     // 控制文本输入框内的文字颜色（必须保证高对比度，清晰可读）
      "focus_keyword": "2到4个字"        // 从作者原文的情境中提炼出的核心情绪词或意象词
    }"""
    try:
        resp = await client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": f"主题：{theme}\n当前文本：{current_text[-200:]}"}],
            temperature=1
        )
        content = resp.choices[0].message.content
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        pass

    # 默认兜底值
    return {
        "page_bg": "#e2e8f0",
        "box_bg": "#f8fafc",
        "text_color": "#334155",
        "focus_keyword": "深渊凝视"
    }

# ==========================================
# WebSocket 路由
# ==========================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            msg_type = payload.get("type")

            # ----------------------------------------------------
            # 组 2 的纯被动自由对话逻辑
            # ----------------------------------------------------
            if msg_type == "group2_chat":
                user_msg = payload.get("text")
                sys_prompt = "你是一个专业的小说创作助手。请根据作者的提问，提供有建设性的建议。尽量简短，不要长篇大论。"

                resp = await client.chat.completions.create(
                    model="kimi-k2.5",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_msg}
                    ],
                    temperature=0.7
                )
                await websocket.send_json({"type": "chat_reply", "content": resp.choices[0].message.content.strip()})

            # ----------------------------------------------------
            # 组 3 的多智能体串行争论逻辑 (小剧场) -> 完整恢复版本
            # ----------------------------------------------------
            elif msg_type == "trigger_theater_intervention":
                theme = payload.get("theme", "")
                text = payload.get("text", "")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 AND门触发！正在拉起串行剧场...")

                # 1. 环境主脑可以和对话并行，因为不影响文字
                env_task = asyncio.create_task(call_env_agent(theme, text))

                # ------ 串行争论开始 ------

                # Act 1: 绝对理性者先发言
                base_context = f"当前设定的世界观主题是：{theme}\n作者目前写了这些内容（可能卡壳了）：\n{text[-300:]}\n"

                sys_rat = "你代表【绝对的理性】。说话必须严谨、冷酷、逻辑缜密（像一个莫得感情的物理学家）。字数控制在40字以内。"
                user_rat = base_context + "\n请一针见血地指出剧情下一步在逻辑或物理法则上的必然走向。"

                res_rat = await call_agent("rational", sys_rat, user_rat)
                # 算出一个发一个，实现真实的“正在输入”的错落感
                await websocket.send_json(
                    {"type": "theater_actor_msg", "role": res_rat["role"], "content": res_rat["content"]})
                await asyncio.sleep(1.5)  # 制造人类打字的停顿感

                # Act 2: 人文关怀者看到理性者的话，出来反驳
                sys_hum = "你代表【人文关怀】。说话必须温柔、充满悲悯、感性（像一个充满哲思的诗人）。字数控制在40字以内。"
                user_hum = base_context + f"\n就在刚刚，【绝对理性者】提出了这个冷酷的建议：\n“{res_rat['content']}”\n\n请你反驳他！从人物内心情感、人性或道德困境的角度，给出更有温度的剧情建议。"

                res_hum = await call_agent("humanist", sys_hum, user_hum)
                await websocket.send_json(
                    {"type": "theater_actor_msg", "role": res_hum["role"], "content": res_hum["content"]})
                await asyncio.sleep(1.5)

                # Act 3: 脑洞者看到前两人的争论，出来嘲讽全场
                sys_cre = "你代表【天马行空的脑洞】。说话必须疯狂、诡异、极具颠覆性甚至有些疯癫。字数控制在40字以内。"
                user_cre = base_context + f"\n刚才，【绝对理性者】说：\n“{res_rat['content']}”\n然后【人文关怀者】反驳道：\n“{res_hum['content']}”\n\n请你嘲笑他们两人的思维太局限、太无聊！抛出一个完全颠覆常理、极具视觉冲击力的疯狂情节转折！"

                res_cre = await call_agent("creative", sys_cre, user_cre)
                await websocket.send_json(
                    {"type": "theater_actor_msg", "role": res_cre["role"], "content": res_cre["content"]})

                # 最后，应用环境改变
                res_env = await env_task
                await websocket.send_json({"type": "env_adjustment", "style": res_env})
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎭 剧场演出完毕！")


    except WebSocketDisconnect:
        active_websockets.remove(websocket)
    except Exception as e:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
        print(f"WS Exception: {e}")

# --- 核心：挂载前端静态资源 ---
@app.get("/")
async def get_index():
    return FileResponse("../frontend/index.html")


app.mount("/", StaticFiles(directory="../frontend"), name="frontend_assets")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)