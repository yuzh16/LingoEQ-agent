import json
from openai import OpenAI
from config import settings

class RouterAgent:
    def __init__(self):
        # 初始化云端大模型客户端
        self.client = OpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL
        )
        self.model_name = settings.LLM_MODEL_NAME
        
        self.system_prompt = """
你是一个顶级的混音意图分析器。你的任务是将用户的口语化音乐制作混音需求转换为结构化的检索标签。 
仅允许输出合法的 JSON 格式，绝不能包含任何 Markdown 标记、解释或其他废话。 

可选的 target_instruments: ["general", "vocal", "drum", "bass", "guitar", "piano", "string", "non_bass"] 
!!!只能使用上述列表中的乐器标签，禁止输出列表外的任何标签！！！ 

输出格式必须严格如下： 
{ "instruments": ["匹配到的乐器1", "匹配到的乐器2"], "keywords": ["听感词汇1", "动作词汇2"] } 

示例： 
用户: "底鼓太糊了，顺便让人声亮一点" 
输出: {"instruments": ["drum", "vocal"], "keywords": ["浑浊", "底鼓", "人声", "明亮", "通透"]}
"""

    def analyze_intent(self, user_input: str) -> dict:
        """
        调用云端顶级大模型进行极速意图拆解
        """
        print(f"☁️ [Router Agent] 正在请求云端满血算力拆解意图: '{user_input}'...")
        
        try:
            # 👈 【核心修改】这里改用 self.client 去请求云端，彻底抛弃 ollama！
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_input}
                ],
                response_format={"type": "json_object"}, # 强制云端返回纯 JSON 格式
                temperature=0.1 # 保持最低温度，确保绝对稳定
            )
            
            # 提取云端返回的结果
            result_str = response.choices[0].message.content
            intent_data = json.loads(result_str)
            
            print(f"🎯 [Router Agent] 拆解成功: {intent_data}")
            return intent_data
            
        except Exception as e:
            print(f"❌ [Router Agent] 云端解析失败，降级回退: {e}")
            return {"instruments": ["general"], "keywords": [user_input]}

# 极速测试代码 (已改回纯同步调用)
if __name__ == "__main__":
    router = RouterAgent()
    # 让我们测试一下满血版 V3 的威力！
    router.analyze_intent("木吉他扫弦感觉太单薄了，想加点厚度，顺便底鼓太糊了")