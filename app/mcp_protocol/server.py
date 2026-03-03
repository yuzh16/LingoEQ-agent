from typing import Dict, Any, List, Optional
import re
from app.services.state_manager import StateManager
from app.core.llm_client import DeepSeekClient
from app.services.rag_engine import RagEngine

class MCPHandler:
    def __init__(self, rag_engine: RagEngine = None):
        print("🛠️ [MCP] 正在初始化各个组件...")
        self.llm = DeepSeekClient()
        self.rag = rag_engine if rag_engine else RagEngine()
        self.state_manager = StateManager(max_history=5)
        print("🧠 [MCP] 状态记忆已就绪")

        # =========================================================
        # 🛡️ 定义允许发给 C++ 的参数白名单 (Sanitizer Whitelist)
        # 必须确保这里的 key 与 EQBand 里的 alias ("freq", "q") 完全一致
        # =========================================================
        self.VALID_TOOL_PARAMS = {
            # 基础参数
            "band_index", "freq", "gain", "q", "filter_type", "enabled", 
            # 动态参数
            "dynamic_enabled", "threshold", "attack", "release",
            # 全局参数
            "output_gain", "bypass"
        }

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理来自 C++ 的 JSON-RPC 请求
        """
        method = request.get("method")
        msg_id = request.get("id")
        
        # 打印请求日志
        if method == "chat/send":
            params = request.get("params", {})
            print(f"📨 [MCP] 收到 Chat 请求: {params.get('content', '')[:30]}...")
        else:
            # 过滤掉高频的心跳和同步日志，避免刷屏
            if method not in ["ping", "eq/sync"]:
                print(f"📨 [MCP] 收到请求: {method}")

        # ----------------------------------------------------
        # 场景 1: 处理用户聊天消息 (核心逻辑)
        # ----------------------------------------------------
        if method == "chat/send":
            user_text = request["params"].get("content", "")
            
            # 1. RAG 检索
            print(f"🔍 [RAG] 正在检索: {user_text}")
            rag_context = self.rag.search(user_text)

            # 2. 获取记忆上下文 (包含当前影子状态、对话历史、以及操作焦点)
            # 这里的 memory_context 已经包含了 "用户刚刚操作了 Band X" 的提示信息
            memory_context = self.state_manager.get_context_prompt()
            
            # 3. LLM 生成
            # 我们完全信任 LLM 能根据 memory_context 中的提示，正确理解“这个”、“那个”指代的是谁
            agent_response = self.llm.generate_response(
                user_input=user_text, 
                rag_context=rag_context,
                conversation_context=memory_context
            )

            if agent_response:
                # 4. 更新状态 (影子状态)
                # 将 LLM 的决策同步到 Python 内存中
                if agent_response.tool_calls:
                    print(f"🔄 [MCP] 同步影子状态 ({len(agent_response.tool_calls)} tools)...")
                    self.state_manager.update_from_tool_calls(agent_response.tool_calls)
                
                reply_content = agent_response.reply_text if agent_response.reply_text else "[执行操作]"
                self.state_manager.add_history(user_text, reply_content)
                
                # 5. 构造 MCP 响应内容列表 (核心重构点：改用影子状态作为数据源)
                content_list: List[Dict[str, Any]] = []

                # Part A: 添加思维链
                if agent_response.thought_process:
                    content_list.append({
                        "type": "text",
                        "role": "thought",
                        "text": agent_response.thought_process
                    })

                # Part B: 添加回复文本
                if agent_response.reply_text:
                    content_list.append({
                        "type": "text",
                        "role": "assistant",
                        "text": agent_response.reply_text
                    })

                # Part C: 添加工具调用 (🛡️ 增强：从影子状态获取清洗后的绝对正确参数)
                for tool in agent_response.tool_calls:
                    if tool.tool_name == "adjust_eq_band":
                        idx = tool.parameters.get("band_index")
                        
                        # 安全检查：确保索引有效
                        if idx is not None and idx in self.state_manager.bands:
                            # --- 核心数据流设计 ---
                            # 1. 从 StateManager 拿到刚刚更新完的、完整的 Band 对象
                            # 2. 使用 by_alias=True 导出，确保 key 是 "freq" 而不是 "frequency"
                            full_band_state = self.state_manager.bands[idx].model_dump(by_alias=True)
                            
                            # 3. 核心清洗逻辑：只保留白名单里的 Key，防止 LLM 输出奇怪的字段
                            clean_params = {
                                k: v for k, v in full_band_state.items() 
                                if k in self.VALID_TOOL_PARAMS
                            }

                            clean_params["band_index"] = idx  # 确保 band_index 始终存在
                            
                            # 发送给 C++
                            content_list.append({
                                "type": "tool_use",
                                "tool": tool.tool_name,
                                "args": clean_params 
                            })
                            
                            print(f"🚀 [Socket] 发送至 C++: Band {idx} | Freq: {clean_params.get('freq')} | Filter: {clean_params.get('filter_type')}")
                        else:
                            print(f"⚠️ [MCP] LLM 输出了无效的频段索引: {idx}")

                # 6. 返回最终的 JSON-RPC 包
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
            try:
                params = request.get("params", {})
                band_idx = params.get("band")
                param_name = params.get("param")
                value = params.get("value")

                if band_idx is not None and param_name:
                    # 更新 Python 端状态
                    # 这里的 update_specific_param 会自动更新 last_manual_time 时间戳
                    # 从而让下一次 Prompt 生成时，能正确反映“用户刚刚动了这个”
                    self.state_manager.update_specific_param(band_idx, param_name, value)
                    return {"jsonrpc": "2.0", "result": "ok", "id": msg_id}
                else:
                    return {"jsonrpc": "2.0", "error": {"code": -32602, "message": "Missing params"}, "id": msg_id}

            except Exception as e:
                print(f"❌ [Sync Error] {e}")
                return {"jsonrpc": "2.0", "error": {"code": -500, "message": str(e)}, "id": msg_id}

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
