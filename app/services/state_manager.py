from typing import List, Dict, Deque, Optional, Any
from collections import deque
import json
import time
from app.schemas.eq_protocol import EQBand, EQFactory, ToolCall

class StateManager:
    """
    状态管理器 (Shadow State)
    职责：
    1. 维护当前 EQ 的全量参数 (24个频段)。
    2. 维护最近的对话历史。
    3. 为 Prompt 提供上下文注入文本（包括操作焦点提示）。
    """
    def __init__(self, max_history: int = 5):
        # 1. 初始化 24 个频段的“影子状态”
        self.bands: Dict[int, EQBand] = {
            i: EQFactory.create_default_band(i) for i in range(24)
        }
        
        # 2. 上下文感知数据 (Context Awareness)
        # 用于追踪“谁”在“什么时候”操作了“哪个频段”
        self.last_llm_band_idx: Optional[int] = None
        self.last_llm_time: float = 0.0
        
        self.last_manual_band_idx: Optional[int] = None
        self.last_manual_time: float = 0.0
        
        # 3. 对话历史
        self.conversation_history: Deque[Dict[str, str]] = deque(maxlen=max_history)

    def update_from_tool_calls(self, tool_calls: List[ToolCall]):
        """
        核心逻辑：根据 LLM 的工具调用，更新 Python 端的影子状态。
        """
        for tool in tool_calls:
            if tool.tool_name == "adjust_eq_band":
                params = tool.parameters
                idx = params.get("band_index")
                
                if idx is not None and idx in self.bands:
                    # 使用 by_alias=True 导出，确保 key 是 "freq"
                    current_band_dict = self.bands[idx].model_dump(by_alias=True)
                    current_band_dict.update(params)
                    
                    try:
                        self.bands[idx] = EQBand.model_validate(current_band_dict)
                        
                        # ✅ 更新 LLM 操作记录
                        self.last_llm_band_idx = idx
                        self.last_llm_time = time.time()

                        print(f"🔄 [State] Band {idx} Updated by AI")
                    
                    except Exception as e:
                        print(f"❌ [State Error] Band {idx} update failed: {e}")

    def update_specific_param(self, band_idx: int, param_name: str, value: Any):
        """
        反向同步专用：处理用户手动操作
        """
        if band_idx not in self.bands:
            return

        band = self.bands[band_idx]
        try:
            # 参数映射与类型转换
            if param_name in ["freq", "frequency"]:
                band.frequency = float(value)
            elif param_name in ["gain"]:
                band.gain = float(value)
            elif param_name in ["q", "q_factor"]:
                band.q_factor = float(value)
            elif param_name in ["type", "filter_type"]:
                band.filter_type = str(value)
            elif param_name in ["bypass", "enabled"]:
                val_bool = bool(float(value)) if isinstance(value, (int, float)) else bool(value)
                if param_name == "bypass":
                    band.enabled = not val_bool
                else:
                    band.enabled = val_bool
            elif param_name in ["dynamic", "dynamic_enabled"]:
                band.dynamic_enabled = bool(value)
            elif param_name in ["threshold"]:
                band.threshold = float(value)
            elif param_name in ["attack"]:
                band.attack = float(value)
            elif param_name in ["release"]:
                band.release = float(value)

            # ✅ 更新手动操作记录
            self.last_manual_band_idx = band_idx
            self.last_manual_time = time.time()

            print(f"🔄 [Sync] Band {band_idx} {param_name} -> {value} (Manual)")
            
        except Exception as e:
            print(f"❌ [State Error] Sync failed: {band_idx}.{param_name}={value} ({e})")

    def add_history(self, user_text: str, assistant_text: str):
        self.conversation_history.append({"role": "user", "content": user_text})
        self.conversation_history.append({"role": "assistant", "content": assistant_text})

    def get_context_prompt(self) -> str:
        """
        生成注入到 System Prompt 中的上下文文本。
        【修改点】去掉 60s 阈值限制，直接陈述客观事实，让 LLM 自己判断时效性。
        """
        current_time = time.time()
        context_blocks = []
        
      # --- Block 1: 操作焦点 (事实陈述) ---
        focus_lines = []
        
        # 1. 描述手动操作事实 (加重语气)
        if self.last_manual_band_idx is not None:
            delta = current_time - self.last_manual_time
            if delta < 120: # 2分钟内都算
                time_str = f"{int(delta)}秒前"
                # 🔥 核心修改：加上 ★★★ 标记，并明确这是用户的物理操作
                focus_lines.append(f"★★★ 关键事实：用户【{time_str}】在宿主里手动触摸了 Band {self.last_manual_band_idx}。")

        # 2. 描述 AI 操作事实
        if self.last_llm_band_idx is not None:
            delta = current_time - self.last_llm_time
            if delta < 120:
                time_str = f"{int(delta)}秒前"
                focus_lines.append(f"- 参考信息：你(AI)在【{time_str}】调节过 Band {self.last_llm_band_idx}。")
        
        if focus_lines:
            context_blocks.append(
                "【⚠️ 视觉焦点与操作记录】\n" + 
                "\n".join(focus_lines) + 
                "\n(判定规则：当用户说“我刚才动的”、“这个”、“手动调的”时，**必须**优先锁定上述带 ★★★ 的频段，忽略你之前的操作记录。)"
            )

        # --- Block 2: EQ 状态快照与空闲频段 (Inventory Management) ---
        active_bands_desc = []
        used_indices = set() # 记录已经被使用的频段 ID
        
        for i in range(24):
            band = self.bands[i]
            
            # 如果频段被 bypass 了，我们暂时不去动它，也算作被占用/不可用
            if not band.enabled: 
                used_indices.add(i)
                continue
            
            # 💡 核心判定：什么叫“被占用”？
            # 只要增益不是 0，或者不是默认的 Peaking，或者开了动态，就算作“被占用”
            is_active = (abs(band.gain) > 0.1) or \
                        (band.filter_type not in ["Peaking"]) or \
                        (band.dynamic_enabled)
            
            if is_active:
                used_indices.add(i) # 标记为已占用
                desc = f"- Band {i}: {band.frequency}Hz, {band.gain:+.1f}dB, {band.filter_type}"
                if band.dynamic_enabled:
                    desc += " [动态开启]"
                active_bands_desc.append(desc)
        
        # 💡 计算空闲频段：24个里面，除去已占用的，剩下的就是空闲的
        free_indices = [i for i in range(24) if i not in used_indices]
        
        # 开始组装文本递给大模型
        state_text = "【当前 EQ 状态 (已占用频段)】\n"
        if active_bands_desc:
            state_text += "\n".join(active_bands_desc) + "\n"
        else:
            state_text += "(暂无，所有频段均为默认平直状态)\n"
            
        # 🔥 最关键的一句：大声告诉它空闲的有哪些！(为了节省 Token，只给它前 5 个可用的就行)
        if free_indices:
            state_text += f"\n【🟢 可用空闲频段 (Free Bands)】: {free_indices[:5]} (当需要新建调整时，请务必从这里挑选 band_index！)"
        else:
            state_text += f"\n【🔴 警告】: 24个频段已全部用满！无法再新建，只能修改现有频段。"

        context_blocks.append(state_text)

        # --- Block 3: 对话历史 (保持不变) ---
        history_text = "【对话历史】\n"
        if not self.conversation_history:
            history_text += "(无)"
        else:
            for msg in self.conversation_history:
                role = "User" if msg['role'] == "user" else "AI"
                history_text += f"{role}: {msg['content']}\n"
        context_blocks.append(history_text)

        return "\n\n".join(context_blocks)