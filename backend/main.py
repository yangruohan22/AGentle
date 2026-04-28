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
import subprocess

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
experiment_path = os.path.join(current_dir, "experiment_data")
app.mount("/experiment_data", StaticFiles(directory=experiment_path), name="experiment_data")
# ==========================================
# 大模型 (LLM) 引擎配置
# ==========================================
DEEPSEEK_API_KEY = "sk-4d52b8a640c445ae95ff13220b0c579d"
client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

active_websockets = []
active_inference_process = None
active_baseline_process = None
active_base_means_process = None  # 🌟 新增：追踪后台基线计算进程
current_baseline_sub_id = None  # 🌟 新增：记住当前正在做基线的被试ID


class InferenceRequest(BaseModel):
    sub_id: str


class SetupData(BaseModel):
    sub_id: str
    group: int
    duration: int = 180


@app.post("/api/start_baseline")
async def start_baseline(data: SetupData):
    global active_baseline_process, current_baseline_sub_id
    current_baseline_sub_id = data.sub_id
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 开始 3 分钟打字基线数据采集 (Sub: {data.sub_id})")
    try:
        # 如果还有上一个没死透的基线进程，先清理掉
        if active_baseline_process is not None:
            active_baseline_process.terminate()

        active_baseline_process = subprocess.Popen(["python", "baseline_recorder.py", data.sub_id, str(data.duration)])
    except Exception as e:
        print(f"启动录制脚本失败: {e}")
    return {"status": "success"}


# ================= 🌟 新增：供前端隐式调用的后台计算接口 =================
@app.post("/api/generate_base_means")
async def generate_base_means():
    global active_baseline_process, active_base_means_process, current_baseline_sub_id

    if not current_baseline_sub_id:
        return {"status": "error", "msg": "找不到当前被试的 Sub_ID"}

    # 安全锁：确保录制脚本已经完全退出，文件写入完毕，防止文件被锁！
    if active_baseline_process is not None:
        if active_baseline_process.poll() is None:
            print(f"⏳ 正在等待录制脚本优雅保存数据并退出...")
            try:
                active_baseline_process.wait(timeout=5)  # 最多等5秒钟让它存文件
            except subprocess.TimeoutExpired:
                active_baseline_process.terminate()
        active_baseline_process = None

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️ 正在后台隐式计算基线特征与提取 ICA (Sub: {current_baseline_sub_id})...")
    try:
        cwd_path = os.path.abspath(os.path.dirname(__file__))
        # 启动计算子进程（不阻塞前端）
        active_base_means_process = subprocess.Popen(["python", "generate_base_means.py", current_baseline_sub_id],
                                                     cwd=cwd_path)
    except Exception as e:
        print(f"⚠️ 自动触发基线计算失败: {e}")

    return {"status": "calculating"}


# ================= 🌟 改造：供前端 ICA 页面轮询的接口 =================
@app.get("/api/check_baseline_status")
async def check_baseline_status():
    global active_base_means_process

    if active_base_means_process is None:
        return {"status": "idle"}

    # 轮询计算进程，而不是录制进程
    ret_code = active_base_means_process.poll()
    if ret_code is None:
        return {"status": "running"}
    else:
        # 计算彻底结束，前端可以拿 ICA 图了
        active_base_means_process = None
        return {"status": "done"}


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


# ================= 一键启动推断引擎 =================
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


# ================= 分类保存实验数据 =================
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


@app.post("/api/clear_alert")
async def clear_alert():
    # 纯转发：告诉所有前端，心流恢复了！解除干预状态
    for ws in active_websockets:
        try:
            await ws.send_json({"type": "clear_alert"})
        except Exception as e:
            print(f"WebSocket发送解除警报失败: {e}")
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
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1,
            extra_body={"thinking": {"type": "disabled"}}
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
            model="deepseek-v4-pro",
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": f"主题：{theme}\n当前文本：{current_text[-200:]}"}],
            temperature=1,
            extra_body={"thinking": {"type": "disabled"}}
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
        "focus_keyword": "觉醒"
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
                # 获取前端传来的历史记录
                history = payload.get("history", [])
                sys_prompt = "你是一个专业的小说创作助手。请根据作者的提问，提供有建设性的建议。尽量简短，不要长篇大论。"

                # 🌟 核心组装：把系统提示词和历史记录拼在一起
                messages = [{"role": "system", "content": sys_prompt}]

                for msg in history:
                    # 前端发来的角色是 'user' 或 'ai_assistant'，API需要 'user' 或 'assistant'
                    api_role = "user" if msg.get("role") == "user" else "assistant"
                    messages.append({"role": api_role, "content": msg.get("content", "")})

                resp = await client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    temperature=1.0,
                    extra_body={"thinking": {"type": "disabled"}}
                )
                await websocket.send_json({"type": "chat_reply", "content": resp.choices[0].message.content.strip()})
            # ----------------------------------------------------
            # 组 3 的多智能体串行争论逻辑 (小剧场) -> 完整恢复版本
            # ----------------------------------------------------
            elif msg_type == "trigger_theater_intervention":
                theme = payload.get("theme", "")
                text = payload.get("text", "")
                history = payload.get("history", [])
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 AND门触发！正在拉起串行剧场...")

                env_task = asyncio.create_task(call_env_agent(theme, text))

                # ==================== 🌟 1. 动态判断：有无文本破冰 🌟 ====================
                text_clean = text.strip()
                if not text_clean:
                    author_status = "【当前状态】：作者正对着空白屏幕发呆，一个字都还没写出来！急需一个极具画面感的【开局第一幕】来破冰。"
                else:
                    author_status = f"【当前进展】：作者目前写了这些内容，但似乎卡壳了：\n“{text[-300:]}”"

                # ==================== 🌟 2. 剧场记忆与对话感规则 🌟 ====================
                history_text = ""
                if history:
                    history_text = "\n【前情提要（你们之前出的主意）】：\n"
                    for msg in history:
                        role_name = {"rational": "理科编剧", "humanist": "感性编剧", "creative": "疯批编剧"}.get(
                            msg.get("role"), "其他")
                        history_text += f"- {role_name}: {msg.get('content')}\n"
                    history_text += "⚠️ 警告：绝对不要重复你们之前说过的情节！\n"

                base_context = f"当前的世界观设定是：{theme}\n{author_status}\n{history_text}"
                base_rule = "【剧场表演法则】：\n1. 必须以【会议室对话的口吻】发言！你可以直接吐槽另外两人的烂主意，或者对作者喊话（如：“听我说作者...”）。\n2. 第一句必须是口语化的感叹或反驳，紧接着立刻扔出一个极度具体的【视觉画面、主角动作或突发事件】来推动剧情。拒绝抽象概念！\n3. 严格控制在60字以内！"

                # ==================== 🌟 3. 演员逐一登场 🌟 ====================

                # 演员 1：绝对理性者
                sys_rat = "你是剧场里的【冷酷理科编剧】。说话一针见血，只关心物理法则、残酷现实和逻辑推演。"
                user_rat = base_context + f"\n{base_rule}\n请你第一个发言。用一两句口语，给主角安排一个具体的危机或环境异变。"

                res_rat = await call_agent("rational", sys_rat, user_rat)
                await websocket.send_json(
                    {"type": "theater_actor_msg", "role": res_rat["role"], "content": res_rat["content"]})
                await asyncio.sleep(1.5)

                # 演员 2：人文关怀者
                sys_hum = "你是剧场里的【感性文艺编剧】。在乎角色的痛苦和人性微光，觉得理科编剧太冷血。"
                user_hum = base_context + f"\n刚刚，理科编剧冷酷地说：\n“{res_rat['content']}”\n\n{base_rule}\n请直接开口反驳理科编剧的冷血！然后从主角的【内心情感或道德抉择】切入，补充一个带有人情味的具体画面。"

                res_hum = await call_agent("humanist", sys_hum, user_hum)
                await websocket.send_json(
                    {"type": "theater_actor_msg", "role": res_hum["role"], "content": res_hum["content"]})
                await asyncio.sleep(1.5)

                # 演员 3：天马行空者
                sys_cre = "你是剧场里的【疯批视觉系编剧】。嫌弃前面两个人太老套、太墨迹。满脑子都是怪诞美学和反直觉的视觉冲击。"
                user_cre = base_context + f"\n理科编剧说：\n“{res_rat['content']}”\n感性编剧反驳说：\n“{res_hum['content']}”\n\n{base_rule}\n请直接开口嘲笑他们俩太无聊！抛开他们的套路，硬塞进来一个完全颠覆常理、极度诡异但又符合设定的绝妙画面！切忌毫无关联的词强行绑定在一起，你所说的虽然天马行空，但是一定要有其内在的逻辑关联。"

                res_cre = await call_agent("creative", sys_cre, user_cre)
                await websocket.send_json(
                    {"type": "theater_actor_msg", "role": res_cre["role"], "content": res_cre["content"]})

                # ====================================================================

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