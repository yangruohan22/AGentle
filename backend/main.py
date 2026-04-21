import asyncio
import json
import re
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

app = FastAPI()

# 允许跨域（本地测试必备）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 核心大模型配置与接口
# ==========================================
KIMI_API_KEY = "sk-KnNJv6ikxTJV6R2VIXdN1SJ7LBZpzBEW0YcVkHqUTuvclgxB"
client = AsyncOpenAI(
    api_key=KIMI_API_KEY,
    base_url="https://api.moonshot.cn/v1",
)


async def call_llm_agent(system_prompt: str, is_json: bool = False):
    """独立的 LLM 调用封装"""
    try:
        response = await client.chat.completions.create(
            model="kimi-k2.5",
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.8
        )
        content = response.choices[0].message.content

        if is_json:
            # 正则过滤，确保提取到合法的 JSON 对象
            match = re.search(r'\{.*\}', content, re.DOTALL)
            return json.loads(match.group(0)) if match else json.loads(content)

        return content.strip()
    except Exception as e:
        print(f"API 调用异常: {e}")
        return {"bg_color": "#1e1e2f"} if is_json else "系统信号干扰中..."


# ==========================================
# A-Gentle 三路并发路由 (核心逻辑)
# ==========================================
async def route_ui_controller(text: str):
    prompt = f"""你是A-Gentle系统的物理环境调节器。用户当前创作陷入停滞。
    已写文本："{text}"
    请仅输出严格的JSON，包含网页背景颜色(HEX格式)。
    如果是悬疑/压抑/紧张情绪，选择深冷色(如 #1e1e2f 或 #0f172a)。
    如果是温馨/日常/放松情绪，选择柔和暖色(如 #fff7ed 或 #fdf4ff)。
    格式: {{"bg_color": "#hexcode"}}"""
    return await call_llm_agent(prompt, is_json=True)


async def route_agent_a(text: str):
    prompt = f"""你是A-Gentle剧场中的'理性刺客'（Agent A）。你冷酷、犀利，专门寻找人类逻辑的漏洞和陈词滥调。
    用户正在进行创作，目前卡壳了。最新文本："{text}"
    要求：用极度简短、一针见血的一句话（不超过20字），指出这段文本最俗套或最不合理的地方。不要给修改建议，只负责打破他的舒适区。"""
    return await call_llm_agent(prompt, is_json=False)


async def route_agent_b(text: str):
    prompt = f"""你是A-Gentle剧场中的'通感大师'（Agent B）。你负责提供剥离视觉的具身物理感受。
    用户最新文本："{text}"
    要求：抛开画面描写，用一句话（不超过20字）补充一个极度细腻的嗅觉、触觉或内脏感觉（例如：胃酸倒流的灼烧、生锈铁丝的腥味、零下十度的刺痛）。不要提建议，直接白描这种感觉。"""
    return await call_llm_agent(prompt, is_json=False)


# ==========================================
# WebSocket 调度中枢
# ==========================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("实验平台（前端）已连接！")

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)

            pause_time = payload.get("pause_duration", 0)
            current_text = payload.get("text", "")

            # FSM 状态机检测：停顿大于5秒且有实际内容，判定为 State 0
            if pause_time >= 5.0 and len(current_text) > 5:
                print(f"检测到停顿 {pause_time:.1f}s，进入 State 0！触发并发引擎...")

                # 通知前端开始播放加载动画
                await websocket.send_json({"type": "status", "message": "A-Gentle 引擎介入中..."})

                # 并发执行三条大模型请求（极大降低延迟）
                ui_task = route_ui_controller(current_text)
                agent_a_task = route_agent_a(current_text)
                agent_b_task = route_agent_b(current_text)

                ui_res, a_res, b_res = await asyncio.gather(ui_task, agent_a_task, agent_b_task)

                # 打包结果发回前端
                await websocket.send_json({
                    "type": "intervention",
                    "ui": ui_res,
                    "theater": [
                        {"role": "Agent A", "content": a_res},
                        {"role": "Agent B", "content": b_res}
                    ]
                })

    except WebSocketDisconnect:
        print("实验平台（前端）已断开连接。")
    except Exception as e:
        print(f"WebSocket 发生错误: {e}")