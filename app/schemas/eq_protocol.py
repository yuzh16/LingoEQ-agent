from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Literal

# --- 基础数据结构 (保持不变，用于定义 EQ 的具体细节) ---
class EQBand(BaseModel):
    """定义单个 EQ 频段的参数"""
    
    # 1. 频段序号
    index: int = Field(..., description="频段序号，范围 0-23")
    
    # 2. 核心参数 (使用 alias 统一接口名称)
    frequency: float = Field(..., alias="freq", description="中心频率 (Hz)") 
    gain: float = Field(..., description="增益 (dB)")
    q_factor: float = Field(..., alias="q", description="Q值") 
    
    # 3. 滤波器类型
    filter_type: Literal["LowCut", "HighCut", "Peaking", "LowShelf", "HighShelf"] = Field(..., description="滤波器类型")
    enabled: bool = Field(default=True, description="开关")
    
    # 4. 动态 EQ 参数
    dynamic_enabled: bool = Field(default=False, description="动态模式开关")
    threshold: float = Field(default=0.0, description="阈值 (dB)")
    attack: float = Field(default=20.0, description="启动时间")
    release: float = Field(default=100.0, description="释放时间")

    # =========================================================
    # 🔧 配置项放在这里 (缩进要在 class 内部)
    # allow_population_by_field_name 是 Pydantic V1 写法
    # populate_by_name 是 Pydantic V2 写法，建议如下配置：
    # =========================================================
    model_config = {
        "populate_by_name": True,  # 允许使用 frequency 赋值，也允许用 freq 赋值
        "extra": "ignore"          # 忽略多余参数，防止报错
    }

# --- 通用协议层 (架构升级) ---

class ToolCall(BaseModel):
    """
    代表一次具体的工具调用意图。
    具有极高的扩展性：未来你可以添加 compressor, limiter 等任何工具，
    只需要改变 tool_name 和 parameters 的内容，不需要修改此结构。
    """
    tool_name: str = Field(..., description="工具名称，例如 'adjust_equalizer', 'reset_eq'")
    parameters: Dict[str, Any] = Field(..., description="该工具所需的具体参数，以字典形式存储")

class AgentResponse(BaseModel):
    """
    Agent 的最终响应结构 (Standard Agent Protocol)
    """
    # 1. 思维链：保留 R1 的思考能力，但要在 JSON 里体现（如果已被 clean_string 移除则可能为空）
    thought_process: str = Field(default="", description="简短的决策逻辑")
    
    # 2. 聊天能力：Agent 可以直接回复文本，而不调用任何工具
    reply_text: Optional[str] = Field(None, description="回复给用户的自然语言文本")
    
    # 3. 工具能力：支持调用 0 个、1 个或多个工具
    # 例如：它可以同时输出 "adjust_equalizer" 和 "update_ui_color"
    tool_calls: List[ToolCall] = Field(default_factory=list, description="要执行的工具调用列表")

    @field_validator('tool_calls')
    def validate_calls(cls, v):
        # 这里可以加一些基础校验，例如检查 tool_name 是否在允许的白名单里
        return v

class EQFactory:
    """辅助类：用于生成默认状态"""
    @staticmethod
    def create_default_band(index: int) -> EQBand:
        """生成一个默认的平直频段"""
        # 默认频率分布算法 (示例: 20Hz - 20kHz 对数分布)
        import math
        base_freq = 20.0 * math.pow(1000.0, index / 23.0)
        
        return EQBand(
            index=index,
            freq=round(base_freq, 1),
            gain=0.0,
            q=1.0,
            filter_type="Peaking",
            dynamic_enabled=False,
            threshold=0.0,
            attack=20.0,
            release=100.0
        )