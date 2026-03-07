import json
import re
from typing import Optional
from app.schemas.eq_protocol import AgentResponse
from openai import OpenAI, APITimeoutError # 【新增】导入 OpenAI 客户端和超时异常

class DeepSeekClient:
    def __init__(self, model_name="deepseek-ai/DeepSeek-V3", base_url="https://api.siliconflow.cn/v1"):
        self.model_name = model_name
        # 【核心修改点】初始化 OpenAI 客户端，不再拼接 /api/chat
        self.client = OpenAI(
            api_key="sk-qmyxmupolothjswkdfocloaykwupxcygnxraligwrqygchlm", # 你的专属密钥
            base_url=base_url
        )

    def extract_thought_and_clean_json(self, raw_text: str) -> tuple[str, str]:
        """
        分离思维链和 JSON 内容
        """
        # 1. 提取 <think> 标签内的思维过程 (修复了这里的乱码)
        thought_match = re.search(r'<think>(.*?)</think>', raw_text, re.DOTALL)
        thought_content = thought_match.group(1).strip() if thought_match else ""

        # 2. 移除 <think> 标签 (为了后续提取 JSON) (修复了这里的乱码)
        text_without_think = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)

        # 3. 提取 JSON (优先匹配 ```json)
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text_without_think, re.DOTALL)
        if json_match:
            clean_json = json_match.group(1)
        else:
            # 兜底：找最外层的大括号
            brace_match = re.search(r'\{.*\}', text_without_think, re.DOTALL)
            clean_json = brace_match.group(0) if brace_match else text_without_think

        # 4. JSON 格式清洗
        clean_json = re.sub(r'"\s*\\\s*\n\s*"', '', clean_json) # 移除 Python 续行符
        clean_json = re.sub(r',\s*\}', '}', clean_json)        # 移除末尾逗号
        clean_json = re.sub(r',\s*\]', ']', clean_json)
        
        # 5. 术语纠错
        clean_json = clean_json.replace('"HighPass"', '"LowCut"')
        clean_json = clean_json.replace('"LowPass"', '"HighCut"')

        return thought_content, clean_json

    def generate_response(self, user_input: str, rag_context: str = "",state_context: str = "",preset_context: str = "",tool_context: str = "") -> Optional[AgentResponse]:
        """
        通用 Agent 响应生成函数
        """
        
        # --- System Prompt: 定义角色与能力 ---
        system_prompt = """
你是一个专业的混音工程师智能体 (LingoEQ)。你的目标是通过控制音频均衡器 (EQ) 来帮助用户获得更好的听感。

【重要提示：实时状态感知】
用户随时可能在宿主软件 (DAW) 中手动调整 EQ 参数。
你看到的【当前 EQ 状态】已经包含了用户的最新手动操作。
因此，当用户说“保持现状”或“微调一下”时，请务必基于【当前 EQ 状态】中的数值进行记忆。
历史隔离：你只需要执行用户的【最新一条输入】！绝对禁止把历史对话中的要求套用到当前这轮操作中！

【你的核心能力】
1. **聊天模式**：解答音频知识，提供混音建议（填充 `reply_text`）。
2. **工具模式**：操作 EQ 插件参数（填充 `tool_calls`）。

【可用工具(Tools)】
工具 1: "adjust_eq_band" (调整频段)
- 作用：调整指定 Band 的 EQ 参数。
工具 2: "set_global_control" (全局控制)
- 作用：控制整体输出。
工具 3: "save_eq_preset" (保存预设)
- 作用：将当前完整的 24-band EQ 状态保存为一个带有特定名称的预设。
工具 4: "load_eq_preset" (加载预设)
- 作用：通过预设名称加载之前保存的 EQ 预设。

***工具的相关介绍和参数在“工具列表”中有详细说明，请务必严格按照工具定义来生成调用参数！***

【输出规则】
1. **只输出纯净的 JSON**：严禁在 JSON 内部或外部包含任何注释、解释、Markdown 标记 (```json) 或多余的符号。
2. **严禁写注释**：绝对不要在 JSON 中使用 "//" 或 "#" 写注释。参数值必须是纯数字或字符串。
3. **多频段并发操作 (绝对强制)**：如果用户的需求包含多个听感目标（并不一定是两个）（例如“清理低频并增加高频”，或要求制作“电话音”），你**必须**在 `tool_calls` 列表中同时生成**多个** `adjust_eq_band` 对象！一次性完成所有操作，不要只做一半！请确保每个操作占用不同的、当前空闲的 band_index。
4. **禁止** 在 key 中出现括号、中文说明、或任何非 key 的文字。
5. **策略选择**：
   -当用户要求“切掉”、“去除”、“清理”某个频段边缘时，优先使用 "LowCut" 或 "HighCut" 而非 "Peaking"。
   -当用户要求“压一下”、“控制”、“动态衰减”某个具体频率（如“压一下 100Hz”）时，**必须**使用 "Peaking" 滤波器配合动态参数，**严禁**使用 LowCut/HighCut。
6. **区分你我 (非常重要)**：
   - 当用户说“**我**刚才手动调整的”、“**我**弄的” -> 你必须去【最近操作记录】中寻找标有“**用户手动调节**”的 Band。
   - 当用户说“刚才那个”、“**你**刚才弄的” -> 你应该寻找“**你(AI)调节**”的 Band。
   
【JSON 输出示例】

场景1-动态处理：用户说"人声的齿音太刺耳了，帮我压一下" (De-essing)
{
    "thought_process": "用户提到齿音(Sibilance)刺耳，这是动态问题。应该在 7kHz-10kHz 使用动态 EQ 进行衰减，只在齿音出现时生效，以免让整体人声变暗。",
    "reply_text": "已为您设置了动态高频衰减，专门针对刺耳的齿音进行控制。",
    "tool_calls": [
        {
            "tool_name": "adjust_eq_band",
            "parameters": {
                "band_index": 18,
                "freq": 8000.0,
                "gain": -6.0,
                "q": 2.0,
                "filter_type": "Peaking",
                "enabled": true,
                "dynamic_enabled": true, 
                "threshold": -20.0,
                "attack": 5.0,
                "release": 50.0
            }
        }
    ]
}

场景2-多频段并发操作：用户说"把这轨木吉他切一下低频，顺便把高频扫弦的清脆感提起来" (需要同时处理低频和高频)
{
    "thought_process": "用户提出了两个独立需求：1. 切低频，需使用 LowCut 滤波器。2. 提升高频清脆感，需使用 HighShelf。我必须同时调用两次工具，分别占用两个不同的空闲 Band。",
    "reply_text": "没问题，我已经为你切除了多余的低频底噪，同时提升了高频，让吉他扫弦听起来更清脆！",
    "tool_calls": [
        {
            "tool_name": "adjust_eq_band",
            "parameters": {
                "band_index": 3,
                "freq": 80.0,
                "gain": 0.0,
                "q": 1.0,
                "filter_type": "LowCut",
                "enabled": true
            }
        },
        {
            "tool_name": "adjust_eq_band",
            "parameters": {
                "band_index": 16,
                "freq": 10000.0,
                "gain": 3.0,
                "q": 0.7,
                "filter_type": "HighShelf",
                "enabled": true
            }
        }
    ]
}
"""

        # --- 注入 RAG及记忆系统 上下文 ---

        context_block = ""
        if tool_context:
            context_block += "\n ===【工具列表】===(请严格按照以下工具定义来生成调用参数) \n" + tool_context + "\n ======================================= \n"

        if state_context:
            context_block += "\n === 【当前EQ状态】(可以基于此进行调整) === \n" + state_context + "\n ======================================= \n"

        if rag_context:
            context_block += "\n ===【参考混音知识库】===(请优先参考以下技巧来决定参数：) \n" + rag_context + "\n ======================================= \n"
        
        if preset_context:
            context_block += "\n ===【当前预设清单】===(请基于以下预设进行反馈：) \n" + preset_context + "\n ======================================= \n"
        
        full_prompt = system_prompt + "\n\n" + context_block + "\n用户输入: \"" + user_input + "\""
        
        print("🧪 [LLM Payload Length]:", len(full_prompt))
        print("🧪 [User Input Raw]:", repr(user_input))

        try:
            log_msg = f"🧠 [LLM] 正在思考: {user_input}"
            if rag_context:
                log_msg += " (已注入 RAG 知识)"
            print(log_msg)

            # -------------------------------------------------------
            # 【核心修复点】 用 OpenAI SDK 替代原有的 requests.post 裸请求
            # -------------------------------------------------------
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": full_prompt},
                    {"role": "user", "content": user_input}
                ],
                response_format={"type": "json_object"}, # 强制 JSON 模式
                temperature=0.3
            )
            
            # 直接获取返回文本
            raw_text = response.choices[0].message.content
            
            # 新增：打印原始输出，定位 key 污染
            print(f"🧾 [LLM Raw Output]: {raw_text[:500]}")

            # -------------------------------------------------------
            # 【核心修复点 2】 解包思维链 (<think>标签内容) 和 JSON
            # -------------------------------------------------------
            real_thought, clean_json = self.extract_thought_and_clean_json(raw_text)
            
            # -------------------------------------------------------
            # 【核心修复点 3】 智能合并思维链
            # -------------------------------------------------------
            try:
                data_dict = json.loads(clean_json)
                
                # 如果 JSON 里没写思考过程，用提取出来的补全
                if not data_dict.get("thought_process") and real_thought:
                    data_dict["thought_process"] = real_thought
                
                # 容错：确保 tool_calls 存在
                if "tool_calls" not in data_dict or not isinstance(data_dict.get("tool_calls"), list):
                    data_dict["tool_calls"] = []

                #最后转为 Pydantic 对象
                agent_data = AgentResponse.model_validate(data_dict)
                return agent_data

            except json.JSONDecodeError:
                print(f"⚠️ JSON 解析失败，原始清洗结果: {clean_json}")
                # 构造一个错误的响应对象返回，防止程序崩溃
                return AgentResponse(
                    thought_process=real_thought,
                    reply_text="抱歉，我的大脑产生了一些无效数据，请重试。",
                    tool_calls=[]
                )
        
        except APITimeoutError: # 【修改】适配 OpenAI 的超时异常
            print("⏱️ [LLM Timeout] 请求超时，可能是模型卡住或 prompt 太长")
            return None

        except Exception as e:
            print(f"❌ [LLM Error] {e}")
            if 'raw_text' in locals():
                print(f"💡 模型原始返回片段: {raw_text[:200]}...")
            return None 
