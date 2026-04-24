import json
import re
from typing import Optional, List, Dict
from app.schemas.eq_protocol import AgentResponse
from openai import OpenAI, APITimeoutError
from config import settings

class DeepSeekClient:
    """
    【MCP Execution Layer - 执行层】
    职责：
    1. 接收标准化 Messages 列表 (List[Dict])。
    2. 调用 OpenAI/DeepSeek API。
    3. 执行模型特有的清洗逻辑 (Regex Parsing, JSON Cleaning)。
    """
    
    def __init__(self):
        self.model_name = settings.LLM_MODEL_NAME
        self.client = OpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.LLM_BASE_URL
        )

    def extract_thought_and_clean_json(self, raw_text: str) -> tuple[str, str]:
        """
        分离思维链和 JSON 内容 (DeepSeek 特有的输出清洗逻辑)
        """
        # 1. 提取 <think> 标签内的思维过程
        # (修复点：这里之前漏掉了，现在补全)
        thought_match = re.search(r'<think>(.*?)</think>', raw_text, re.DOTALL)
        thought_content = thought_match.group(1).strip() if thought_match else ""

        # 2. 移除 <think> 标签 (为了后续提取 JSON)
        text_without_think = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)

        # 3. 提取 JSON (优先匹配 ```json 代码块)
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text_without_think, re.DOTALL)
        if json_match:
            clean_json = json_match.group(1)
        else:
            # 兜底：找最外层的大括号
            brace_match = re.search(r'\{.*\}', text_without_think, re.DOTALL)
            clean_json = brace_match.group(0) if brace_match else text_without_think

        # 4. JSON 格式清洗 (处理模型常见的语法错误)
        clean_json = re.sub(r'"\s*\\\s*\n\s*"', '', clean_json) # 移除 Python 续行符
        clean_json = re.sub(r',\s*\}', '}', clean_json)        # 移除末尾逗号
        clean_json = re.sub(r',\s*\]', ']', clean_json)
        
        # 5. 术语纠错 (标准化滤波器名称)
        clean_json = clean_json.replace('"HighPass"', '"LowCut"')
        clean_json = clean_json.replace('"LowPass"', '"HighCut"')

        return thought_content, clean_json

    def call_api(self, messages: List[Dict[str, str]]) -> Optional[AgentResponse]:
        """
        核心执行方法：发送请求 -> 获取响应 -> 解析结果
        :param messages: 由 PromptEngineer 渲染好的对话列表
        """
        
        # 打印调试信息，确认 payload 大小
        try:
            total_chars = sum(len(m['content']) for m in messages)
            print(f"🧪 [LLM Payload]: {total_chars} chars | User: {messages[-1]['content'][:50]}...")
        except Exception:
            pass

        try:
            print(f"🧠 [LLM] 正在思考...")

            # -------------------------------------------------------
            # 1. 发起 API 请求 (完全保留你的 OpenAI SDK 调用方式)
            # -------------------------------------------------------
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"}, # 强制 JSON 模式
                temperature=settings.LLM_TEMPERATURE
            )
            
            # 2. 获取原始文本
            raw_text = response.choices[0].message.content
            print(f"🧾 [LLM Raw Output]: {raw_text[:500]}") # 打印前500字用于调试

            # -------------------------------------------------------
            # 3. 解析与清洗
            # -------------------------------------------------------
            real_thought, clean_json = self.extract_thought_and_clean_json(raw_text)
            
            try:
                data_dict = json.loads(clean_json)
                
                # 如果 JSON 里没写思考过程，用提取出来的补全
                if not data_dict.get("thought_process") and real_thought:
                    data_dict["thought_process"] = real_thought
                
                # 容错：确保 tool_calls 列表存在
                if "tool_calls" not in data_dict or not isinstance(data_dict.get("tool_calls"), list):
                    data_dict["tool_calls"] = []

                # 4. 转换为 Pydantic 对象 (强类型校验)
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
        
        except APITimeoutError:
            print("⏱️ [LLM Timeout] 请求超时，可能是模型卡住或 prompt 太长")
            return None

        except Exception as e:
            print(f"❌ [LLM Error] {e}")
            if 'raw_text' in locals():
                print(f"💡 模型原始返回片段: {raw_text[:200]}...")
            return None
