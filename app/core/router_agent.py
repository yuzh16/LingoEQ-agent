import json
import ollama

class RouterAgent:
    def __init__(self, model_name="deepseek-r1:7b"):
        self.model_name = model_name
        # 强约束的 System Prompt：逼迫模型变成一台无情的 JSON 机器
        self.system_prompt = """
你是一个顶级的混音意图分析器。你的任务是将用户的口语化音乐制作混音需求转换为结构化的检索标签。
仅允许输出合法的 JSON 格式，绝不能包含任何 Markdown 标记、解释或其他废话。

可选的 target_instruments: ["general", "vocal", "drum", "bass", "guitar", "piano", "string", "non_bass"]
!!!只能使用上述列表中的乐器标签，禁止输出列表外的任何标签！！！

输出格式必须严格如下：
{
    "instruments": ["匹配到的乐器1", "匹配到的乐器2"],
    "keywords": ["听感词汇1", "动作词汇2"]
}

示例：
用户: "底鼓太糊了，顺便让人声亮一点"
输出: {"instruments": ["drum", "vocal"], "keywords": ["浑浊", "底鼓", "人声", "明亮", "通透"]}
"""

    def analyze_intent(self, user_input: str) -> dict:
        """
        调用本地大模型进行极速意图拆解
        """
        print(f"🧠 [Router Agent] 正在光速拆解意图: '{user_input}'...")
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_input}
                ],
                format="json",  # 👈 这是 Ollama 极重要的参数，强制其走 JSON 输出模式
                options={"temperature": 0.1} # 温度调到最低，拒绝大模型的发散思维，保证绝对稳定
            )
      
            result_str = response['message']['content']
            intent_data = json.loads(result_str)
            print(f"🎯 [Router Agent] 拆解成功: {intent_data}")
            return intent_data
         
        except Exception as e:
            print(f"❌ [Router Agent] 解析失败，降级回退到通用模式: {e}")
            # 容灾机制：如果模型突然抽风，返回保底的 fallback
            return {"instruments": ["general"], "keywords": [user_input]}

# 测试代码
if __name__ == "__main__":
    import asyncio
    router = RouterAgent()
    asyncio.run(router.analyze_intent("木吉他扫弦感觉太单薄了，想加点厚度"))
