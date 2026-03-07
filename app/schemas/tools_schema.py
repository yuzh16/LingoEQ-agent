# app/core/schemas/tools_schema.py

# ==============================================================================
# LingoEQ 工具定义集
# 这里的定义将被转换为 JSON Schema 发送给 DeepSeek，告知它可用的能力。
# ==============================================================================

LINGO_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "adjust_eq_band",
            "description": "调整 EQ 特定频段的参数。解决听感问题（如'低频浑浊'、'高频刺耳'）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "band_index": { "type": "integer", "description": "频段序号 (0-23)。" },
                    "freq": { "type": "number", "description": "中心频率 Hz (20.0 - 20000.0)。" },
                    "gain": { "type": "number", "description": "增益 dB (-24.0 到 +24.0)。正数为提升，负数为衰减。" },
                    "q": { "type": "number", "description": "Q值 (带宽)。0.1 (宽) 到 10.0 (窄)。" },
                    "filter_type": { 
                        "type": "string", 
                        "enum": ["LowCut", "HighCut", "Peaking", "LowShelf", "HighShelf"],
                        "description": "滤波器类型。"
                    },
                    "enabled": { "type": "boolean", "description": "启用/禁用。" },
                    "dynamic_enabled": { "type": "boolean", "description": "是否启用动态 EQ 模式。" },
                    "threshold": { "type": "number", "description": "动态阈值 dB (-60.0 到 0.0)。" },
                    "attack": { "type": "number", "description": "启动时间 ms。" },
                    "release": { "type": "number", "description": "释放时间 ms。" }
                },
                "required": ["band_index", "freq", "gain", "q", "filter_type", "enabled"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_global_control",
            "description": "控制全局输出增益或旁通。",
            "parameters": {
                "type": "object",
                "properties": {
                    "output_gain": { "type": "number", "description": "总输出增益 dB。" },
                    "bypass": { "type": "boolean", "description": "全局旁通开关。" }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_eq_preset",
            "description": "【记忆操作】将当前 EQ 状态保存为预设。当用户说'保存当前音色'、'记住这个设置'时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "preset_name": {
                        "type": "string",
                        "description": "预设名称。例如：'温暖人声', '我的摇滚贝斯'。"
                    }
                },
                "required": ["preset_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "load_eq_preset",
            "description": "【记忆操作】加载已保存的预设。当用户说'加载那个温暖人声'、'用之前的设置'时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "preset_name": {
                        "type": "string",
                        "description": "要查询的预设名称关键词。"
                    }
                },
                "required": ["preset_name"]
            }
        }
    }
]

def get_tools_prompt() -> str:
    """
    辅助函数：将工具定义转换为 System Prompt 友好的字符串。
    如果使用 OpenAI SDK，可以直接传 LINGO_TOOLS；如果用 Prompt 工程，用这个。
    """
    import json
    return f"""
## Available Tools
You can manipulate the audio EQ using the following tools. 
Strictly follow the JSON format when calling them.

{json.dumps(LINGO_TOOLS, indent=2)}
"""
