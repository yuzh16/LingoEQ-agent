from typing import Dict, Any, List
from app.services.state_manager import StateManager
from app.core.llm_client import DeepSeekClient
from app.services.rag_engine import RagEngine
from app.core.router_agent import RouterAgent
from app.services.memory_manager import MemoryManager
from app.schemas.eq_protocol import EQBand
from app.schemas.tools_schema import get_tools_prompt

class MCPHandler:
    def __init__(self, rag_engine: RagEngine = None):
        print("🛠️ [MCP] 正在初始化各个组件...")
        self.llm = DeepSeekClient()
        self.rag = rag_engine if rag_engine else RagEngine()
        self.router = RouterAgent()
        self.state_manager = StateManager(max_history=5)
        self.memory = MemoryManager()
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
            
            # 1-1. 极速意图拆解 (将大白话转为专业乐器标签与听感词)
            print(f"⚡ [Agentic] 阶段一：正在进行意图光速拆解...")
            intent_data = self.router.analyze_intent(user_text)

            # 1-2. 精准知识检索 (利用提纯后的关键词去数据库拿干货)
            print(f"🔍 [Agentic] 阶段二：正在根据意图进行定向 RAG 检索...")
            # 👉 注意：这里调用的是刚刚加进 RagEngine 的新方法！
            rag_context = self.rag.search(intent_data)

            # 2. 获取记忆上下文与预设目录 (为 LLM 提供全局状态和预设的参考)
            state_context = self.state_manager.get_context_prompt()
            preset_context = self.memory.get_all_preset_names()
            tool_context = get_tools_prompt()  # 获取工具使用说明，确保 LLM 知道怎么调用工具
            print(f"🤖 [Agentic] 阶段三：知识注入完毕，正在呼叫主厨大模型生成最终 EQ 动作...")
            
            # 3. LLM 生成
            # 我们完全信任 LLM 能根据 state_context 中的提示，正确理解“这个”、“那个”指代的是谁
            agent_response = self.llm.generate_response(
                user_input=user_text,
                rag_context=rag_context,
                state_context=state_context,
                preset_context=preset_context,
                tool_context=tool_context
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
                    
                    #1.普通调参
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

                    #2.保存预设
                    elif tool.tool_name == "save_eq_preset":
                        preset_name = tool.parameters.get("preset_name", "未命名预设")
                        print(f"💾 [MCP] 主厨调用了【保存预设】: {preset_name}")
                        # Python 物理提取当前 24 频段并存入 ChromaDB
                        self.memory.save_preset(preset_name, self.state_manager)

                    # 3. 加载预设 (修改了这里)
                    elif tool.tool_name == "load_eq_preset":
                        preset_name = tool.parameters.get("preset_name", "未命名预设")
                        print(f"📂 [MCP] 主厨调用了【加载预设】: {preset_name}")
                        
                        preset_state_list, status_msg = self.memory.load_preset(preset_name)
                        
                        if preset_state_list:
                            clean_batch_params = []
                            
                            # 【核心修改点】使用 enumerate 提供兜底索引 i
                            # 防止旧预设数据中缺少 'band_index' 导致报错
                            for i, band_data in enumerate(preset_state_list):
                                
                                # 尝试获取索引，如果拿不到(None)，就用循环计数器 i 代替
                                idx = band_data.get("band_index", i)
                                
                                # 强制把索引写回数据源，确保后续 model_validate 也能读到
                                band_data["band_index"] = idx

                                # 1. 更新 Python 端影子状态
                                # 注意：如果你的 EQBand 模型里必须包含 band_index，现在它安全了
                                self.state_manager.bands[idx] = EQBand.model_validate(band_data)
                                
                                # 2. 数据清洗
                                clean_band = {k: v for k, v in band_data.items() if k in self.VALID_TOOL_PARAMS}
                                clean_band["band_index"] = idx # 再次确保发给 C++ 的包里一定有 index
                                clean_batch_params.append(clean_band)
                            
                            # 3. 组装 C++ 端批量跳跃动画指令
                            content_list.append({
                                "type": "tool_use",
                                "tool": "adjust_eq_batch", 
                                "args": {"bands": clean_batch_params}
                            })
                            print(f"🚀 [Socket] 批量跳跃指令已打包 ({len(clean_batch_params)} bands) -> C++")
                        else:
                            # 如果没找到预设，需要告诉 LLM
                            agent_response.reply_text = status_msg
                            content_list.append({
                                "type": "text",
                                "role": "assistant",
                                "text": status_msg
                            })
                
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
