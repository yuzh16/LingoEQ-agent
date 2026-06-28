"""
lingoeq_bench_server.py — LingoEQ-Bench 评测专用 HTTP Facade 入口

职责：
  把 LingoEQ 内部流水线以一个标准 HTTP 接口暴露出去，
  供 Benchmark 评测框架通过轻量 adapter 调用，无需触碰 TCP socket 层。

启动方式：
  python lingoeq_bench_server.py          # 默认监听 127.0.0.1:8889
  BENCH_PORT=9000 python lingoeq_bench_server.py

接口：
  POST /benchmark
  POST /health

注意：
  - 每次请求创建全新 AgentDispatcher，保证 test case 间状态完全隔离。
  - 本文件与 main.py 完全解耦，不干扰正常生产运行。
  - 需要先安装评测依赖：pip install fastapi uvicorn
"""

import os
import sys
import uvicorn
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 确保项目根目录在 sys.path 中，支持从任意工作目录启动
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.dispatcher.server import AgentDispatcher


# ============================================================
# 请求 / 响应 Schema  (严格对齐 LingoEQ-Bench 标准 payload)
# ============================================================

class LockedBand(BaseModel):
    """用户已手动调节过、需要永久锁定的频段（对应 README state_manager_data.locked_bands）"""
    band_id: int = Field(..., description="频段编号 0-23")
    type: str = Field(default="Peaking", description="滤波器类型，对应 LingoEQ 内部 filter_type")
    freq: float = Field(..., description="中心频率 (Hz)")
    gain: float = Field(..., description="增益 (dB)")
    q: float = Field(..., description="Q 值")
    status: Optional[str] = Field(default=None, description="装饰字段 [USER-LOCKED]，注入逻辑中忽略")

class StateManagerData(BaseModel):
    """对应 README environment_context.state_manager_data"""
    locked_bands: Optional[List[LockedBand]] = Field(default=None)
    # idle_bands 是评测框架声明的合法 band_id 池，LLM 输出的 band_id 必须在此范围内
    idle_bands: Optional[List[int]] = Field(default=None)

class SensoryRawData(BaseModel):
    """
    原始频谱帧数据，由 SensoryManager 进行心理声学处理后注入上下文。
    format: 元数据字段，注入逻辑中忽略
    frequencies_hz: 24 个频段的中心频率 (Hz)
    frames: 20 帧 × 24 频段，每个值 0.0~1.0（归一化幅度）
    """
    format: Optional[str] = Field(default=None, description="元数据，如 time_series_spectrum_20x24，忽略")
    frequencies_hz: List[float]
    frames: List[List[float]]

class EnvironmentContext(BaseModel):
    """评测环境注入数据，模拟真实插件运行时的宿主状态"""
    state_manager_data: Optional[StateManagerData] = Field(default=None)
    sensory_raw_data: Optional[SensoryRawData] = Field(default=None)

class BenchmarkRequest(BaseModel):
    """POST /benchmark 请求体"""
    prompt: str = Field(..., description="用户自然语言指令")
    environment_context: Optional[EnvironmentContext] = Field(default=None, description="环境注入上下文")
    # 消融实验开关：设为 True 时跳过对应模块，用于消融实验
    skip_sensory: bool = Field(default=False, description="消融：跳过音频感知数据注入，原始 frames 以文本注入 prompt")
    skip_state: bool = Field(default=False, description="消融：跳过 EQ 状态上下文注入与物理护栏，原始状态数据以文本注入 prompt")
    skip_rag: bool = Field(default=False, description="消融：跳过 RAG 知识库检索注入")

class EQAction(BaseModel):
    """
    单条 EQ 调参动作。
    六个 README 必填字段 + LingoEQ 内部扩展字段（评估器会忽略多余字段）。
    """
    band_id: int
    freq: float
    gain: float
    q: float
    type: str       # 对应 LingoEQ 内部 filter_type，评估器对大小写不敏感
    enabled: bool
    # 以下为 LingoEQ 扩展字段，README 未要求但无害
    dynamic_enabled: bool
    threshold: float
    attack: float
    release: float

class BenchmarkResponse(BaseModel):
    """POST /benchmark 响应体"""
    actions: List[EQAction] = Field(default_factory=list, description="LLM 产生的 EQ 调参动作列表")


# ============================================================
# FastAPI App + 全局单例 Handler
# ============================================================

app = FastAPI(
    title="LingoEQ Benchmark Facade",
    description="LingoEQ-Bench 评测专用接口，绕过 TCP socket 直接调用内部流水线",
    version="1.0.0",
    docs_url="/docs"        # 访问 http://127.0.0.1:8889/docs 可查看交互式文档
)

# 服务启动时初始化一次，embedding 模型和 ChromaDB 只加载一次
# 每次请求通过 reset() 清空状态，避免重复初始化带来的巨大开销
_handler = AgentDispatcher()


# ============================================================
# 核心辅助函数
# ============================================================

def _inject_environment(handler: AgentDispatcher, env: Optional[EnvironmentContext],
                        skip_state: bool = False, skip_sensory: bool = False) -> None:
    """
    将 Benchmark 提供的环境数据注入到已 reset 的 handler 中。

    skip_state=True：跳过全部 StateManager 注入（含锁定频段和虚拟锁定），物理护栏完全不生效。
                     原始数据由 _build_raw_context() 文本化追加到 prompt，LLM 自行解读。
    skip_sensory=True：跳过 SensoryManager 注入，原始 frames 由 _build_raw_context() 文本化追加。
    """
    if env is None:
        return

    sm = handler.state_manager

    if not skip_state and env.state_manager_data:
        smd = env.state_manager_data
        # 1. 注入用户锁定频段（触发 StateManager 永久锁定机制）
        if smd.locked_bands:
            for band in smd.locked_bands:
                sm.update_specific_param(band.band_id, "freq",        band.freq)
                sm.update_specific_param(band.band_id, "gain",        band.gain)
                sm.update_specific_param(band.band_id, "q",           band.q)
                sm.update_specific_param(band.band_id, "filter_type", band.type)
            print(f"🔒 [Bench] 已注入并锁定 {len(smd.locked_bands)} 个用户频段")

        # 2. 虚拟锁定非 idle 频段
        if smd.idle_bands is not None:
            idle_set = set(smd.idle_bands)
            locked_set = set(b.band_id for b in smd.locked_bands) if smd.locked_bands else set()
            for i in range(24):
                if i not in idle_set and i not in locked_set:
                    sm.virtual_lock_registry[i] = True
            print(f"🔒 [Bench] idle_bands={list(idle_set)}，其余频段已虚拟锁定")

    # 3. 注入音频感知快照
    if not skip_sensory and env.sensory_raw_data:
        raw = env.sensory_raw_data
        handler.sensory_manager.process_and_add_snapshot(
            frames=raw.frames,
            frequencies=raw.frequencies_hz,
            slot_index=0
        )
        print("👂 [Bench] 已注入音频感知快照 (slot 0)")


def _build_raw_context(env: Optional[EnvironmentContext], skip_state: bool, skip_sensory: bool) -> str:
    """
    消融模式下，将被跳过的模块对应的原始数据序列化为文本，追加到 prompt 末尾。
    保证四组消融实验的信息量完全一致，唯一变量是处理模块的存在与否。
    """
    import json
    if env is None:
        return ""
    parts = []
    if skip_state and env.state_manager_data:
        parts.append("state_manager_data: " + json.dumps(env.state_manager_data.model_dump(), ensure_ascii=False))
    if skip_sensory and env.sensory_raw_data:
        parts.append("sensory_raw_data: " + json.dumps(env.sensory_raw_data.model_dump(), ensure_ascii=False))
    return ("\n\n" + "\n\n".join(parts)) if parts else ""


def _extract_actions(content_list: List[Dict[str, Any]]) -> List[EQAction]:
    """
    从 _build_response_content() 返回的列表中提取 adjust_eq_band 动作，
    并将字段名映射为 Benchmark 标准格式（band_index→band_id, filter_type→type）。
    """
    actions = []
    for item in content_list:
        if item.get("type") != "tool_use":
            continue
        if item.get("tool") != "adjust_eq_band":
            continue

        args: Dict[str, Any] = item.get("args", {})
        actions.append(EQAction(
            band_id=args.get("band_index", -1),
            freq=float(args.get("freq", 0.0)),
            gain=float(args.get("gain", 0.0)),
            q=float(args.get("q", 1.0)),
            type=str(args.get("filter_type", "Peaking")),
            enabled=bool(args.get("enabled", True)),
            dynamic_enabled=bool(args.get("dynamic_enabled", False)),
            threshold=float(args.get("threshold", 0.0)),
            attack=float(args.get("attack", 20.0)),
            release=float(args.get("release", 100.0)),
        ))
    return actions


# ============================================================
# 路由
# ============================================================

@app.get("/health")
def health():
    """心跳检测，供 Benchmark runner 在调用前确认服务就绪"""
    return {"status": "ok", "service": "lingoeq-bench-facade"}


@app.post("/benchmark", response_model=BenchmarkResponse)
def benchmark(request: BenchmarkRequest):
    """
    主评测接口。

    流程：
      1. 创建全新 AgentDispatcher（每次请求独立实例，保证 test case 状态隔离）
      2. 注入环境数据（locked_bands + sensory_raw_data）
      3. 执行内部流水线（跳过 TCP socket 层）
      4. 提取 adjust_eq_band 动作并映射字段名
      5. 返回标准 Benchmark 响应
    """
    print(f"\n{'='*60}")
    print(f"📋 [Bench] 收到评测请求: {request.prompt[:60]}...")

    # Step 1：复用全局单例，只重置状态（避免重复加载 embedding 模型和 ChromaDB）
    _handler.state_manager.reset()
    _handler.sensory_manager.reset()
    _handler.current_reflexion_depth = 0

    # Step 2：注入环境
    _inject_environment(_handler, request.environment_context,
                        skip_state=request.skip_state,
                        skip_sensory=request.skip_sensory)

    # 消融模式：被跳过模块的原始数据直接文本化追加到 prompt，保持信息量一致
    effective_prompt = request.prompt + _build_raw_context(
        request.environment_context,
        skip_state=request.skip_state,
        skip_sensory=request.skip_sensory
    )

    # Step 3：内部流水线（与 AgentDispatcher.process_request chat/send 分支逻辑对齐）
    try:
        context_data = _handler.context_manager.collect_resources(
            effective_prompt,
            skip_sensory=request.skip_sensory,
            skip_state=request.skip_state,
            skip_rag=request.skip_rag
        )
        messages = _handler.prompt_engineer.render_payload(context_data)
        agent_response = _handler.llm.call_api(messages)
    except Exception as e:
        print(f"❌ [Bench] 流水线执行异常: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    if agent_response is None:
        print("❌ [Bench] LLM 返回 None，无法提取动作")
        raise HTTPException(status_code=502, detail="LLM returned no response")

    # 同步影子状态（保证 _build_response_content 拿到的是经过 guardrail 处理后的参数）
    if agent_response.tool_calls:
        _handler.state_manager.update_from_tool_calls(agent_response.tool_calls)

    # Step 4：序列化并提取 adjust_eq_band 动作
    content_list = _handler._build_response_content(agent_response)
    actions = _extract_actions(content_list)

    print(f"✅ [Bench] 流水线完成，提取到 {len(actions)} 条 EQ 动作")
    print(f"{'='*60}\n")

    return BenchmarkResponse(actions=actions)


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    port = int(os.getenv("BENCH_PORT", "8889"))
    host = os.getenv("BENCH_HOST", "127.0.0.1")

    print(f"🚀 [LingoEQ-Bench] Facade 服务启动中...")
    print(f"   监听地址: http://{host}:{port}")
    print(f"   接口文档: http://{host}:{port}/docs")
    print(f"   主评测端点: POST http://{host}:{port}/benchmark")

    uvicorn.run(app, host=host, port=port, log_level="warning")
