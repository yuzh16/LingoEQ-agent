import math
import time
from typing import List, Dict, Deque, Optional, Any
from collections import deque
from app.schemas.eq_protocol import EQBand, EQFactory, ToolCall
from config import settings

class StateManager:
    """
    状态管理器 (Shadow State)
    集成 Neuro-Symbolic 架构：认知层引导 (Prompt) + 物理层护栏 (DSP logic)
    """
    def __init__(self, max_history: int = 5):
        # 1. 初始化 24 个频段的“影子状态”
        self.bands: Dict[int, EQBand] = {
            i: EQFactory.create_default_band(i) for i in range(24)
        }
        
        # 2. 上下文感知数据
        self.last_llm_band_idx: Optional[int] = None
        self.last_llm_time: float = 0.0
        self.last_manual_band_idx: Optional[int] = None
        self.last_manual_time: float = 0.0
        
        # 3. 对话历史（用于 Prompt 注入）
        self.conversation_history: Deque[Dict[str, str]] = deque(maxlen=max_history)

        # ==========================================
        # 🛡️ 显式逻辑配置 (Guardrail Config)
        # ==========================================
        # 只要用户动过，该频段 ID 永久锁定，AI 不得直操
        self.manual_touch_registry: Dict[int, bool] = {} 
        
        # 声学保护系数：保护范围覆盖带宽，提供平滑的干涉缓冲
        self.SAFETY_MARGIN = settings.SAFETY_MARGIN       

    def _get_free_bands(self) -> List[int]:
        """
        获取当前未被占用的空闲频段列表。
        判定标准：1. 未被用户手动操作过；2. 增益为0且为默认 Peaking 类型且未开启动态。
        """
        free_indices = []
        for i in range(24):
            # 如果 ID 在手动保护名单里，绝对不空闲
            if i in self.manual_touch_registry:
                continue
            
            band = self.bands[i]
            # 检查该频段是否已有 AI 激活的操作
            is_active = (abs(band.gain) > 0.1) or \
                        (band.filter_type not in ["Peaking"]) or \
                        (band.dynamic_enabled)
            
            if not is_active:
                free_indices.append(i)
                
        return free_indices

    def _calculate_gain_attenuation(self, llm_f: float, llm_q: float, hum_f: float, hum_q: float) -> float:
        """
        [核心算法] 基于重叠尺度的削减算法。
        根据 DSP 滤波器带宽 (BW ≈ 1/Q) 计算两个频点的相互影响指数。
        """
        # 容错处理：确保频率和Q值在合法数学域
        llm_f, hum_f = max(20.0, llm_f), max(20.0, hum_f)
        llm_q, hum_q = max(0.1, llm_q), max(0.1, hum_q)

        # 1. 计算两个频点的绝对距离（八度 Octave）
        d_oct = abs(math.log2(llm_f / hum_f))
        
        # 2. 计算干涉临界阈值
        # 滤波器的影响半径近似为带宽的一半：R = 1/(2Q)
        r_llm = 1.0 / (2.0 * llm_q)
        r_hum = 1.0 / (2.0 * hum_q)
        threshold = (r_llm + r_hum) * self.SAFETY_MARGIN
        
        # 3. 计算保留比例
        if d_oct >= threshold:
            return 1.0  # 互不干扰，保留 100% Gain
        
        # 重叠率公式：距离越近，重叠率(0~1)越高
        overlap_ratio = 1.0 - (d_oct / threshold)
        
        # 返回保留比例：重叠越多，保留越少。
        # 例如完全同频时，overlap=1.0, retention=0.0 (增益被完全抹除)
        return 1.0 - overlap_ratio
    
    # 在 StateManager.py 中
    def _apply_safety_constraints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """内部私有方法：执行物理底线钳位"""
        # 定义物理极限 (可根据需要调准)
        LIMITS = {
            "gain": (settings.GAIN_LIMIT_MIN, settings.GAIN_LIMIT_MAX),
            "freq": (settings.FREQ_LIMIT_MIN, settings.FREQ_LIMIT_MAX),
            "frequency": (settings.FREQ_LIMIT_MIN, settings.FREQ_LIMIT_MAX),
            "q": (settings.Q_LIMIT_MIN, settings.Q_LIMIT_MAX),
            "q_factor": (settings.Q_LIMIT_MIN, settings.Q_LIMIT_MAX)
        }
    
        for key, (min_val, max_val) in LIMITS.items():
            if key in params:
                raw_val = float(params[key])
                clamped_val = max(min_val, min(max_val, raw_val))
                if clamped_val != raw_val:
                    print(f"🛡️ [Physical Guardrail] {key}: {raw_val} -> {clamped_val}")
                    params[key] = clamped_val
        return params


    def update_specific_param(self, band_idx: int, param_name: str, value: Any):
        """
        同步层：当用户在宿主软件上手动调节时调用。
        此操作会触发 ID 永久锁定。
        """
        if band_idx not in self.bands:
            return

        band = self.bands[band_idx]
        try:
            # 属性映射逻辑
            if param_name in ["freq", "frequency"]: band.frequency = float(value)
            elif param_name in ["gain"]: band.gain = float(value)
            elif param_name in ["q", "q_factor"]: band.q_factor = float(value)
            elif param_name in ["type", "filter_type"]: band.filter_type = str(value)
            elif param_name in ["bypass", "enabled"]:
                val_bool = bool(float(value)) if isinstance(value, (int, float)) else bool(value)
                band.enabled = not val_bool if param_name == "bypass" else val_bool
            elif param_name in ["dynamic", "dynamic_enabled"]: band.dynamic_enabled = bool(value)
            
            # ✅ 永久锁定该频段 ID
            self.manual_touch_registry[band_idx] = True
            self.last_manual_band_idx = band_idx
            self.last_manual_time = time.time()

            print(f"🔄 [Manual Sync] Band {band_idx} {param_name} -> {value}. ID 永久锁定。")
            
        except Exception as e:
            print(f"❌ [Sync Error] {e}")

    def update_from_tool_calls(self, tool_calls: List[ToolCall]):
        """
        执行层：处理 LLM 生成的参数调整。
        包含：重定向逻辑、干涉削减算法、满载丢弃保护。
        """
        current_time = time.time()
        
        for tool in tool_calls:
            if tool.tool_name == "adjust_eq_band":
                params = tool.parameters
                idx = params.get("band_index")
                
                if idx is None or idx not in self.bands:
                    continue

                # --- 🧱 显式逻辑 1：ID 永久锁定与重定向 ---
                if idx in self.manual_touch_registry:
                    new_idx = self._get_nearest_free_band(idx)
                    if new_idx is not None:
                        print(f"🔀 [Guardrail] Band {idx} 受用户保护。自动重定向至最近空闲频段 Band {new_idx}")
                        idx = new_idx
                        params["band_index"] = idx
                    else:
                        print(f"🚫 [Guardrail] 24频段已满且 Band {idx} 锁定，无处安置，操作丢弃。")
                        continue

                # --- 🌫️ 显式逻辑 2：Q-Aware 频域互相影响削减 ---
                if "gain" in params:
                    # 确定 AI 拟操作的物理属性
                    llm_f = float(params.get("freq", params.get("frequency", self.bands[idx].frequency)))
                    llm_q = float(params.get("q", params.get("q_factor", self.bands[idx].q_factor)))
                    
                    min_retention = 1.0
                    interfered_by = None
                    
                    # 遍历所有手动操作过的频段，计算干涉
                    for m_idx in self.manual_touch_registry.keys():
                        if m_idx == idx: continue
                        
                        hum_f = self.bands[m_idx].frequency
                        hum_q = self.bands[m_idx].q_factor
                        
                        retention = self._calculate_gain_attenuation(llm_f, llm_q, hum_f, hum_q)
                        if retention < min_retention:
                            min_retention = retention
                            interfered_by = m_idx
                    
                    # 应用削减
                    if min_retention < 1.0:
                        original_gain = params["gain"]
                        params["gain"] = original_gain * min_retention
                        print(f"📉 [DSP Guardrail] 靠近用户频段(Band {interfered_by})。重叠削减: {original_gain:+.2f} -> {params['gain']:+.2f}")

                params = self._apply_safety_constraints(params)

                # --- 最终状态落库 ---
                current_band_dict = self.bands[idx].model_dump(by_alias=True)
                current_band_dict.update(params)
                
                try:
                    self.bands[idx] = EQBand.model_validate(current_band_dict)
                    self.last_llm_band_idx = idx
                    self.last_llm_time = current_time
                except Exception as e:
                    print(f"❌ [State Validation Error] {e}")

    def _get_nearest_free_band(self, target_idx: int) -> Optional[int]:
        """返回距离 target_idx 编号最近的空闲频段，无空闲时返回 None。"""
        free = self._get_free_bands()
        if not free:
            return None
        return min(free, key=lambda i: abs(i - target_idx))

    def apply_preset_with_guardrails(self, preset_bands: list) -> list:
        """
        预设加载的受保护写入入口，行为与 update_from_tool_calls 对齐：
        - 锁定频段自动重定向到最近空闲频段；无空闲则丢弃。
        - 对所有写入执行安全钳位。
        返回实际被写入的频段字典列表，供调用方序列化发往 C++。
        """
        applied = []
        for i, band_data in enumerate(preset_bands):
            band_data = dict(band_data)
            idx = band_data.get("band_index", i)  # enumerate 兜底：预设数据缺字段时按顺序落库

            if idx not in self.bands:
                continue

            if idx in self.manual_touch_registry:
                nearest = self._get_nearest_free_band(idx)
                if nearest is not None:
                    print(f"🔀 [Preset Guardrail] Band {idx} 被用户锁定，重定向至最近空闲频段 Band {nearest}")
                    idx = nearest
                else:
                    print(f"🚫 [Preset Guardrail] Band {idx} 锁定且无空闲频段，跳过。")
                    continue

            band_data["band_index"] = idx
            band_data["index"] = idx
            band_data = self._apply_safety_constraints(band_data)

            try:
                self.bands[idx] = EQBand.model_validate(band_data)
                applied.append(band_data)
            except Exception as e:
                print(f"❌ [Preset Load Error] Band {idx}: {e}")

        return applied

    def add_history(self, user_text: str, assistant_text: str):
        """记录对话，供上下文感知使用"""
        self.conversation_history.append({"role": "user", "content": user_text})
        self.conversation_history.append({"role": "assistant", "content": assistant_text})

    def get_context_prompt(self, skip: bool = False) -> str:
        """
        生成注入 System Prompt 的上下文，实现”人机认知同步”。
        skip=True 时返回空字符串，用于消融实验或不需要状态上下文的场景。
        """
        if skip:
            return ""

        current_time = time.time()
        context_blocks = []

        # 1. 焦点提示
        focus_lines = []
        if self.last_manual_band_idx is not None:
            focus_lines.append(f"★ 锁定提示：用户手动调节过 Band {self.last_manual_band_idx}。系统禁止 AI 修改该 ID，且会在物理层拦截任何靠近该频点的干扰操作。")

        if self.last_llm_band_idx is not None:
            delta = int(current_time - self.last_llm_time)
            focus_lines.append(f"- 操作回顾：你(AI)在 {delta} 秒前调节过 Band {self.last_llm_band_idx}。")

        if focus_lines:
            context_blocks.append("【⚠️ 物理约束与操作同步】\n" + "\n".join(focus_lines))

        # 2. 状态快照
        active_bands_desc = []
        for i in range(24):
            band = self.bands[i]
            if i in self.manual_touch_registry or (abs(band.gain) > 0.1) or (band.filter_type != "Peaking"):
                lock_tag = "[USER-LOCKED]" if i in self.manual_touch_registry else ""
                active_bands_desc.append(f"- Band {i}: {band.frequency:.0f}Hz, {band.gain:+.1f}dB, Q={band.q_factor:.2f} {lock_tag}")

        free_indices = self._get_free_bands()
        state_text = "【当前 EQ 资源状态】\n" + ("\n".join(active_bands_desc) if active_bands_desc else "(空闲)")
        state_text += f"\n\n🟢 可用空闲 Band ID: {free_indices[:5]}" if free_indices else f"\n\n🔴 警告：所有 Band 已占满。"
        context_blocks.append(state_text)

        # 3. 历史对话
        history_text = "【近期交互记录】\n" + "\n".join([f"{m['role'].upper()}: {m['content']}" for m in self.conversation_history])
        context_blocks.append(history_text)

        return "\n\n".join(context_blocks)
