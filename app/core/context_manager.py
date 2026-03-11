from typing import Dict, Any
from app.services.state_manager import StateManager
from app.services.memory_manager import MemoryManager
from app.services.rag_engine import RagEngine
from app.core.router_agent import RouterAgent
from app.services.sensory_manager import SensoryManager
from app.schemas.tools_schema import get_tools_prompt

class ContextManager:
    """
    【MCP Resource Layer - 资源层】
    职责：
    1. 接收用户原始输入。
    2. 调度 Router 分析意图。
    3. 并行/串行收集 RAG、State、Memory 等资源。
    4. 输出结构化的 Context 字典 (Resource Dictionary)，不包含任何 System Prompt 模板拼接逻辑。
    """

    def __init__(self, 
                 state_manager: StateManager,
                 memory_manager: MemoryManager,
                 rag_engine: RagEngine,
                 router_agent: RouterAgent,
                 sensory_manager: SensoryManager):
        self.state = state_manager
        self.memory = memory_manager
        self.rag = rag_engine
        self.router = router_agent
        self.sensory = sensory_manager
        print("📦 [ContextManager] 资源管理器已就绪")

    def collect_resources(self, user_input: str) -> Dict[str, Any]:
        """
        核心方法：收集所有上下文资源
        """
        print(f"📦 [ContextManager] 正在聚合上下文资源...")

        # 1. 意图分析 (Router)
        # 必须先执行，因为 RAG 依赖意图数据
        print(f"⚡ [ContextManager] 正在请求意图分析...")
        intent_data = self.router.analyze_intent(user_input)

        # 2. 知识检索 (RAG)
        # 将结构化的 intent 传给 RAG 引擎
        print(f"🔍 [ContextManager] 正在检索混音知识...")
        rag_content = self.rag.search(intent_data)

        # 3. 状态快照 (Shadow State)
        # 获取当前 EQ 参数、最近手动操作记录、对话历史
        state_content = self.state.get_context_prompt()

        # 4. 记忆检索 (Memory/Presets)
        # 获取预设列表
        preset_content = self.memory.get_all_preset_names()

        # 5. 工具定义 (Tools)
        # 获取工具的 Schema 描述
        tool_definitions = get_tools_prompt()

        # 6. 音频感知数据 (Sensory Data)
        sensory_content = self.sensory.get_sensory_data()

        # 7. 打包资源
        # 返回纯数据字典，不做字符串拼接
        resources = {
            "user_input": user_input,
            "intent": intent_data,         # 保留意图原始数据，供 PromptEngineer 决策使用
            "resources": {
                "rag_context": rag_content,       # 对应原 rag_context
                "state_context": state_content,   # 对应原 state_context
                "preset_context": preset_content, # 对应原 preset_context
                "tool_context": tool_definitions,  # 对应原 tool_context
                "sensory_context": sensory_content # 对应原 sensory_context
            }
        }
        
        print(f"📦 [ContextManager] 资源聚合完成。")
        return resources
