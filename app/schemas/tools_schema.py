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
            "description": "Adjust parameters for a specific EQ band filter. Use this to fix tonal issues (e.g., 'muddy bass', 'harsh highs').",
            "parameters": {
                "type": "object",
                "properties": {
                    "band_index": {
                        "type": "integer",
                        "description": "The band number (0-23). 0-2: Sub-bass, 3-6: Bass, 7-15: Mids, 16-23: Highs.",
                        "minimum": 0,
                        "maximum": 23
                    },
                    "freq": {
                        "type": "number",
                        "description": "Center frequency in Hz (20.0 - 20000.0)."
                    },
                    "gain": {
                        "type": "number",
                        "description": "Gain in dB (-24.0 to +24.0). Positive to Boost, Negative to Cut."
                    },
                    "q": {
                        "type": "number",
                        "description": "Q factor (Bandwidth). 0.1 (Wide) to 10.0 (Narrow/Surgical)."
                    },
                    "filter_type": {
                        "type": "string",
                        "description": "The shape/type of the filter. Use 'Peaking' for general tone shaping, 'LowCut'/'HighCut' to remove rumbling/hissing.",
                        "enum": ["LowCut", "HighCut", "Peaking", "LowShelf", "HighShelf"] 
                    },
                    "enabled": {
                        "type": "boolean",
                        "description": "Enable or disable this band."
                    },
                    "dynamic_enabled": {
                        "type": "boolean",
                        "description": "Set to true to enable Dynamic EQ mode (compress/expand based on input level)."
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Dynamic threshold in dB (-60.0 to 0.0). When signal exceeds this, gain is applied dynamically."
                    },
                    "attack": {
                        "type": "number",
                        "description": "Attack time in ms (0.1 to 100.0). How fast the dynamic EQ reacts."
                    },
                    "release": {
                        "type": "number",
                        "description": "Release time in ms (10.0 to 2000.0). How fast it returns to normal."
                    }
                },
                "required": ["band_index"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_global_control",
            "description": "Control global output settings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "output_gain": {
                        "type": "number",
                        "description": "Master Output Gain in dB."
                    },
                    "bypass": {
                        "type": "boolean",
                        "description": "Global Bypass."
                    }
                }
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
