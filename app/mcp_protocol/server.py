from typing import Dict, Any, List
from app.services.state_manager import StateManager
from app.services.memory_manager import MemoryManager
from app.services.rag_engine import RagEngine
from app.core.router_agent import RouterAgent
from config import settings

# --- 新架构核心组件 ---
from app.core.context_manager import ContextManager
from app.core.prompt_engineer import PromptEngineer
from app.services.sensory_manager import SensoryManager
from app.core.llm_client import DeepSeekClient

class MCPHandler:
    """
    【MCP Dispatcher Layer - 调度层】
    职责：
    1. 接收 C++ 的 JSON-RPC 请求。
    2. 协调 ContextManager -> PromptEngineer -> DeepSeekClient 的流水线执行。
    3. 处理 Tool Calls 的后处理逻辑 (参数清洗、预设加载兜底)。
    4. 返回标准 JSON-RPC 响应。
    """
    def __init__(self, rag_engine: RagEngine = None):
        print("🛠️ [MCP] 正在初始化系统组件...")
        
        # 1. 基础服务层 (Services)
        self.state_manager = StateManager(max_history=5)
        self.memory = MemoryManager()
        self.rag = rag_engine if rag_engine else RagEngine()
        self.router = RouterAgent()
        self.sensory_manager = SensoryManager()
        
        # 2. 核心架构层 (Core Architecture)
        # 依赖注入：ContextManager 需要访问所有基础服务
        self.context_manager = ContextManager(
            state_manager=self.state_manager,
            memory_manager=self.memory,
            rag_engine=self.rag,
            router_agent=self.router,
            sensory_manager=self.sensory_manager
        )
        self.prompt_engineer = PromptEngineer()
        self.llm = DeepSeekClient()

        # [新增] 闭环验证深度控制 (防止死循环)
        self.MAX_REFLEXION_DEPTH = settings.MAX_REFLEXION_DEPTH  # 允许 AI 自我纠错的最大次数
        self.current_reflexion_depth = 0
        self.is_auto_check_enabled = True  # 👉 【新增】记录当前是否开启了自检模式
        
        print("🧠 [MCP] 架构重构完成：Dispatcher 已就绪")

        # =========================================================
        # 🛡️ 定义允许发给 C++ 的参数白名单 (Sanitizer Whitelist)
        # =========================================================
        self.VALID_TOOL_PARAMS = {
            "band_index", "freq", "gain", "q", "filter_type", "enabled", 
            "dynamic_enabled", "threshold", "attack", "release",
            "output_gain", "bypass"
        }

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理来自 C++ 的 JSON-RPC 请求
        """
        method = request.get("method")
        msg_id = request.get("id")
        
        # 日志过滤
        if method == "chat/send":
            params = request.get("params", {})
            print(f"📨 [MCP] 收到 Chat 请求: {params.get('content', '')[:30]}...")
        elif method not in ["ping", "eq/sync"]:
            print(f"📨 [MCP] 收到请求: {method}")

        # ----------------------------------------------------
        # 场景 1: 处理用户聊天消息 (重构后的流水线)
        # ----------------------------------------------------
        if method == "chat/send":
            user_text = request["params"].get("content", "")
            
            # 👉 【新增】读取 C++ 传来的开关状态（默认为 True 防崩溃）
            self.is_auto_check_enabled = params.get("auto_check", True)
            print(f"🎛️ [UI 控制] 本次对话是否开启自检: {self.is_auto_check_enabled}")
            # [新增] 用户发起新对话，重置验证深度
            self.current_reflexion_depth = 0

            # === Pipeline Start ===
            
            # Step 1: 资源聚合 (Resource Layer)
            # 输出：包含 intent, rag_context, state_context 的纯数据字典
            context_data = self.context_manager.collect_resources(user_text)

            # Step 2: 提示词渲染 (Presentation Layer)
            # 输出：OpenAI 格式的 messages 列表
            messages = self.prompt_engineer.render_payload(context_data)

            # Step 3: 执行推理 (Execution Layer)
            # 输出：结构化的 AgentResponse 对象
            agent_response = self.llm.call_api(messages)
            
            # === Pipeline End ===

            if agent_response:
                # 4. 更新状态 (影子状态同步)
                if agent_response.tool_calls:
                    print(f"🔄 [MCP] 同步影子状态 ({len(agent_response.tool_calls)} tools)...")
                    self.state_manager.update_from_tool_calls(agent_response.tool_calls)
                
                reply_content = agent_response.reply_text if agent_response.reply_text else "[执行操作]"
                self.state_manager.add_history(user_text, reply_content)
                
                # 5. 构造 MCP 响应 (Tool Execution & Serialization)
                content_list = self._build_response_content(agent_response)
                
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "role": "assistant",
                        "content": content_list
                    }
                }
            else:
                return {
                    "jsonrpc": "2.0", 
                    "id": msg_id, 
                    "error": {"code": -500, "message": "LLM 生成失败或格式错误"}
                }
        
        # ----------------------------------------------------
        # 场景 2: 参数反向同步 (C++ -> Python)
        # ----------------------------------------------------
        elif method == "eq/sync":
            return self._handle_sync(request, msg_id)
        
        # ----------------------------------------------------
        # 场景 2.5: [修改] 接收音频感知快照并存入指定槽位 (C++ -> Python)
        # ----------------------------------------------------
        elif method == "audio/snapshot_upload":
            params = request.get("params", {})
            frames = params.get("frames", [])
            frequencies = params.get("frequencies", [])
            slot_index = params.get("slot_index")  # [新增] 提取槽位 ID
            
            if frames and frequencies and slot_index is not None:
                # 把数据和槽位 ID 一起扔给 SensoryManager 处理
                self.sensory_manager.process_and_add_snapshot(frames, frequencies, slot_index)
                return {
                    "jsonrpc": "2.0", 
                    "id": msg_id,
                    "result": f"snapshot_received_in_slot_{slot_index}"
                }
            else:
                return {
                    "jsonrpc": "2.0", 
                    "id": msg_id,
                    "error": {"code": -32602, "message": "Missing audio frames or slot_index"}
                }

        # ----------------------------------------------------
        # 场景 2.6: [新增] 精确删除音频感知快照 (C++ -> Python)
        # ----------------------------------------------------
        elif method == "audio/snapshot_delete":
            params = request.get("params", {})
            slot_index = params.get("slot_index")
            
            if slot_index is not None:
                self.sensory_manager.delete_snapshot(slot_index)
                return {
                    "jsonrpc": "2.0", 
                    "id": msg_id,
                    "result": f"snapshot_deleted_slot_{slot_index}"
                }
            else:
                return {
                    "jsonrpc": "2.0", 
                    "id": msg_id,
                    "error": {"code": -32602, "message": "Missing slot_index"}
                }

        # ----------------------------------------------------
        # 场景 2.7: [新增] 接收影子录音机的 A/B 对比数据并触发 LLM 反思
        # ----------------------------------------------------
        elif method == "audio/reflexion_diff_upload":
            params = request.get("params", {})
            pre_frames = params.get("pre_frames", [])
            post_frames = params.get("post_frames", [])
            frequencies = params.get("frequencies", [])

            if not pre_frames or not post_frames:
                return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32602, "message": "Missing diff frames"}}

            # 1. 深度计数器累加
            self.current_reflexion_depth += 1
            print(f"🧠 [MCP] 接收到闭环对比数据，正在唤醒 LLM 进行二次反思 (当前深度: {self.current_reflexion_depth}/{self.MAX_REFLEXION_DEPTH})...")

            # 2. 调用感知层生成报告
            diff_report = self.sensory_manager.process_reflexion_diff(pre_frames, post_frames, frequencies)

            # 3. 【核心拦截】：如果达到最大深度，强制修改报告，下达“最终通牒”
            if self.current_reflexion_depth >= self.MAX_REFLEXION_DEPTH:
                diff_report += "\n\n【⚠️ 系统强制警告】你已经达到了最大闭环验证次数！" \
                               "为了保护音频引擎，你现在被**绝对禁止**再次调用任何 adjust_eq 工具。 " \
                               "你必须立刻向用户输出一段文本，总结你刚才尝试了哪些操作、目前的效果如何，并请用户手动听取或介入调整。"

            # === 二次 Pipeline Start ===
            context_data = self.context_manager.collect_resources(
                user_input=diff_report,
                skip_rag=True
            )
            messages = self.prompt_engineer.render_payload(context_data)
            agent_response = self.llm.call_api(messages)
            # === 二次 Pipeline End ===

            if agent_response:
                # 4. 【终极保险】：就算大模型叛逆了，我们也在物理层把它截断
                if self.current_reflexion_depth >= self.MAX_REFLEXION_DEPTH and agent_response.tool_calls:
                    print("🛡️ [Safety Net] LLM 试图违抗指令继续调用工具，系统已强制拦截。")
                    agent_response.tool_calls = [] # 强行清空它的工具调用
                    if not agent_response.reply_text:
                        agent_response.reply_text = "我已经进行了多次闭环尝试，为了避免过度处理，我已停止自动调整。请您检查当前音色是否满意。"

                # 5. 【核心路由】：根据是否合格，决定发给 C++ 气泡还是状态码
                if agent_response.tool_calls:
                    # 【情况 B：验证不合格，触发二段微调】
                    print(f"🔄 [MCP] LLM 验收未通过，决定继续微调 ({len(agent_response.tool_calls)} tools)...")
                    self.state_manager.update_from_tool_calls(agent_response.tool_calls)
                    
                    # 兜底：如果模型没说话，补一句简略说明
                    if not agent_response.reply_text:
                        agent_response.reply_text = "检查发现仍有频段冲突，已进行二次微调修正。"
                        
                    self.state_manager.add_history("【系统声学对比报告】已送达", agent_response.reply_text)
                    content_list = self._build_response_content(agent_response)
                else:
                    # 【情况 A：验证合格，没有动工具】
                    print("✅ [MCP] LLM 验收通过，物理状态完美。")
                    self.state_manager.add_history("【系统声学对比报告】已送达", "闭环验证通过，无明显突刺。")
                    
                    # ⚠️ 关键：直接伪造一个隐藏的 status 信号，不生成任何聊天气泡
                    content_list = [{
                        "type": "status",
                        "status_code": "success",
                        "message": "自我检查合格"
                    }]

                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "role": "assistant",
                        "content": content_list
                    }
                }
            else:
                return {
                    "jsonrpc": "2.0", 
                    "id": msg_id, 
                    "error": {"code": -500, "message": "LLM 闭环反思失败"}
                }

        # ----------------------------------------------------
        # 场景 3: 心跳 (Ping)
        # ----------------------------------------------------
        elif method == "ping":
            return {"jsonrpc": "2.0", "id": msg_id, "result": "pong"}

        else:
            return {
                "jsonrpc": "2.0", 
                "id": msg_id, 
                "error": {"code": -32601, "message": "Method not found"}
            }

    def _build_response_content(self, agent_response) -> List[Dict[str, Any]]:
        """
        私有辅助方法：将 AgentResponse 转换为 C++ 可读的 JSON 列表
        包含工具执行的具体逻辑 (Sanitization, Batching 等)
        """
        content_list: List[Dict[str, Any]] = []
        has_eq_action = any(t.tool_name in ["adjust_eq_band", "adjust_eq_batch"] for t in agent_response.tool_calls)
       
        # Part A: 思维链
        # 【修改】只有在首轮盲调时（深度为0），才展示 Thinking 气泡
        if agent_response.thought_process and self.current_reflexion_depth == 0:
            content_list.append({
                "type": "text",
                "role": "thought",
                "text": agent_response.thought_process
            })

        # Part B: 回复文本
        if agent_response.reply_text:
            # 👉 【修改】第一轮正常显示为 assistant，第二轮（自我修正）标记为 correction
            role_type = "assistant" if self.current_reflexion_depth == 0 else "correction"
            
            content_list.append({
                "type": "text",
                "role": role_type,
                "text": agent_response.reply_text
            })

        # 告诉 C++ UI 该显示什么悬浮动画
        if has_eq_action and getattr(self, 'is_auto_check_enabled', True):
            msg = "⚙️ LingoEQ 自我检查中..." if self.current_reflexion_depth == 0 else "🔬 进行二段精调检查中..."
            content_list.append({
                "type": "status",
                "status_code": "checking",
                "message": msg
            })
            
        # Part C: 工具执行与参数清洗
        for tool in agent_response.tool_calls:
            
            # 1. 普通调参 (adjust_eq_band)
            if tool.tool_name == "adjust_eq_band":
                idx = tool.parameters.get("band_index")
                if idx is not None and idx in self.state_manager.bands:
                    # 从影子状态获取清洗后的绝对正确参数
                    full_band_state = self.state_manager.bands[idx].model_dump(by_alias=True)
                    clean_params = {
                        k: v for k, v in full_band_state.items() 
                        if k in self.VALID_TOOL_PARAMS
                    }
                    clean_params["band_index"] = idx
                    
                    content_list.append({
                        "type": "tool_use",
                        "tool": tool.tool_name,
                        "args": clean_params 
                    })
                    print(f"🚀 [Socket] 发送至 C++: Band {idx} | Freq: {clean_params.get('freq')}")
                else:
                    print(f"⚠️ [MCP] LLM 输出了无效的频段索引: {idx}")

            # 2. 保存预设 (save_eq_preset)
            elif tool.tool_name == "save_eq_preset":
                preset_name = tool.parameters.get("preset_name", "未命名预设")
                print(f"💾 [MCP] 主厨调用了【保存预设】: {preset_name}")
                self.memory.save_preset(preset_name, self.state_manager)

            # 3. 加载预设 (load_eq_preset)
            elif tool.tool_name == "load_eq_preset":
                preset_name = tool.parameters.get("preset_name", "未命名预设")
                print(f"📂 [MCP] 主厨调用了【加载预设】: {preset_name}")
                
                preset_state_list, status_msg = self.memory.load_preset(preset_name)
                
                if preset_state_list:
                    applied_bands = self.state_manager.apply_preset_with_guardrails(preset_state_list)

                    if applied_bands:
                        clean_batch_params = [
                            {**{k: v for k, v in b.items() if k in self.VALID_TOOL_PARAMS}, "band_index": b["band_index"]}
                            for b in applied_bands
                        ]
                        content_list.append({
                            "type": "tool_use",
                            "tool": "adjust_eq_batch",
                            "args": {"bands": clean_batch_params}
                        })
                        print(f"🚀 [Socket] 批量预设指令 -> C++ ({len(clean_batch_params)} bands)")
                    else:
                        content_list.append({
                            "type": "text",
                            "role": "assistant",
                            "text": "所有频段均被用户手动锁定，预设未能应用。请先解除手动调节。"
                        })
                else:
                    # 没找到预设
                    if not agent_response.reply_text: # 如果 LLM 没说话，帮它补一句
                        content_list.append({
                            "type": "text",
                            "role": "assistant",
                            "text": status_msg
                        })
                
        return content_list

    def _handle_sync(self, request, msg_id):
        """处理 C++ -> Python 的状态同步"""
        try:
            params = request.get("params", {})
            band_idx = params.get("band")
            param_name = params.get("param")
            value = params.get("value")

            if band_idx is not None and param_name:
                self.state_manager.update_specific_param(band_idx, param_name, value)
                return {"jsonrpc": "2.0", "result": "ok", "id": msg_id}
            else:
                return {"jsonrpc": "2.0", "error": {"code": -32602, "message": "Missing params"}, "id": msg_id}

        except Exception as e:
            print(f"❌ [Sync Error] {e}")
            return {"jsonrpc": "2.0", "error": {"code": -500, "message": str(e)}, "id": msg_id}
