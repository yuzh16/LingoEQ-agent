# app/services/sensory_manager.py
from collections import deque
from datetime import datetime
from typing import List

class SensoryManager:
    def __init__(self):
        # 使用 deque 自动维护最大 5 个槽位的 FIFO 队列
        self.snapshots = deque(maxlen=5)

    def process_and_add_snapshot(self, frames: List[List[float]], frequencies: List[float]):
        """处理接收到的 20 帧原始数据，提取特征并存入槽位"""
        if not frames or not frequencies:
            return

        num_frames = len(frames)
        num_bands = len(frequencies)
        
        # 1. 计算均值 (Mean) 和 峰值 (Max)
        avg_spectrum = [0.0] * num_bands
        max_spectrum = [0.0] * num_bands
        
        for frame in frames:
            for i in range(num_bands):
                if i < len(frame):
                    val = frame[i]
                    avg_spectrum[i] += val
                    if val > max_spectrum[i]:
                        max_spectrum[i] = val
                        
        avg_spectrum = [val / num_frames for val in avg_spectrum]

        # 2. 整体频段平衡分析 (三频能量比)
        # 假设 24 频段：0-7(低频), 8-15(中频), 16-23(高频)
        low_energy = sum(avg_spectrum[0:8]) / 8 if num_bands >= 8 else 0
        mid_energy = sum(avg_spectrum[8:16]) / 8 if num_bands >= 16 else 0
        high_energy = sum(avg_spectrum[16:24]) / 8 if num_bands == 24 else 0

        # 3. 提取共振点 (Top Peaks) 和 缺失点 (Bottom Valleys)
        band_energies = [{"freq": frequencies[i], "energy": avg_spectrum[i]} for i in range(num_bands)]
        sorted_bands = sorted(band_energies, key=lambda x: x["energy"])
        
        # 排除完全没声音的死区(energy < 0.01)来找有效缺失点
        valleys = [b for b in sorted_bands if b["energy"] > 0.01][:2] 
        peaks = sorted_bands[-3:] # 能量最高的 3 个点
        peaks.reverse()

        # 4. 动态特征分析 (瞬态/稳态)
        # 比较 Max 和 Avg 的差距。如果差距很大，说明是打击乐/跳跃的声音
        crest_factors = [(max_spectrum[i] - avg_spectrum[i]) for i in range(num_bands)]
        avg_crest = sum(crest_factors) / num_bands
        dynamic_type = "瞬态极强 (如打击乐/拨弦)" if avg_crest > 0.3 else "相对平稳 (如人声/铺底)"

        # 记录快照
        snapshot = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "balance": {"low": low_energy, "mid": mid_energy, "high": high_energy},
            "peaks": peaks,
            "valleys": valleys,
            "dynamic_type": dynamic_type
        }
        self.snapshots.append(snapshot)
        print(f"👂 [SensoryManager] 深度听觉特征分析完成，当前记忆槽位: {len(self.snapshots)}/5")

    def get_sensory_data(self) -> str:
        """将存储的频谱特征翻译为给 LLM 的 System Prompt 上下文"""
        if not self.snapshots:
            return "当前没有用户提供的音频采样参考。"

        context_lines = ["\n以下数据是通过对 1 秒钟音频快照分析得出的物理特征，能量范围为 0.0(极弱) 到 1.0(极强)。"]
        
        for i, snap in enumerate(self.snapshots):
            bal = snap['balance']
            bal_str = f"低频区={bal['low']:.2f}, 中频区={bal['mid']:.2f}, 高频区={bal['high']:.2f}"
            peaks_str = ", ".join([f"{p['freq']:.1f}Hz({p['energy']:.2f})" for p in snap["peaks"]])
            valleys_str = ", ".join([f"{v['freq']:.1f}Hz({v['energy']:.2f})" for v in snap["valleys"]])
            
            line = f"""
--- 采样 #{i+1} ({snap['timestamp']}) ---
1. 整体平衡: {bal_str}
2. 动态特性: {snap['dynamic_type']}
3. 能量堆积区 (可能导致浑浊/轰鸣/刺耳): {peaks_str}
4. 能量缺失区 (相对较弱，如需增加空气感/厚度可在此提升): {valleys_str if valleys_str else '无明显缺失'}"""
            context_lines.append(line)
            
        return "\n".join(context_lines)