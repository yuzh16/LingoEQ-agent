# app/services/sensory_manager.py
from datetime import datetime
from typing import List, Dict
import numpy as np
from config import settings

class SensoryManager:
    def __init__(self):
        # 精确管理 0~4 号槽位
        self.snapshots: Dict[int, dict] = {}
        self.silent_snapshot: dict = {}

    def _get_a_weighting(self, freq: float) -> float:
        """A计权 (IEC 61672:2003)，0dB 参考点在 1kHz。"""
        if freq <= 0:
            return -100.0
        f_sq = freq ** 2
        ra = (12194**2 * f_sq**2) / ((f_sq + 20.6**2) * np.sqrt((f_sq + 107.7**2) * (f_sq + 737.9**2)) * (f_sq + 12194**2))
        return 20 * np.log10(ra) + 2.0

    def _get_k_weighting(self, freq: float) -> float:
        """K计权 (ITU-R BS.1770) 频率响应采样，0dB 参考点在 1kHz。
        通过对两级数字滤波器（预滤波器 + RLB 高通）的频率响应采样实现逐频段计权。
        """
        fs = 48000.0
        w = 2 * np.pi * freq / fs
        z = np.exp(1j * w)

        # Stage 1: 高频搁架预滤波器 (~+4dB above 1kHz)
        b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
        a1 = [1.0, -1.69065929318241, 0.73248077421585]
        H1 = (b1[0] + b1[1]/z + b1[2]/z**2) / (a1[0] + a1[1]/z + a1[2]/z**2)

        # Stage 2: RLB 高通滤波器 (截止 ~38Hz)
        b2 = [1.0, -2.0, 1.0]
        a2 = [1.0, -1.99004745483398, 0.99007225036621]
        H2 = (b2[0] + b2[1]/z + b2[2]/z**2) / (a2[0] + a2[1]/z + a2[2]/z**2)

        db = 20 * np.log10(abs(H1 * H2) + 1e-12)

        # 归一化：在 1kHz 处为 0dB
        w_ref = 2 * np.pi * 1000.0 / fs
        z_ref = np.exp(1j * w_ref)
        H1r = (b1[0] + b1[1]/z_ref + b1[2]/z_ref**2) / (a1[0] + a1[1]/z_ref + a1[2]/z_ref**2)
        H2r = (b2[0] + b2[1]/z_ref + b2[2]/z_ref**2) / (a2[0] + a2[1]/z_ref + a2[2]/z_ref**2)
        db_ref = 20 * np.log10(abs(H1r * H2r) + 1e-12)

        return db - db_ref

    def _get_iso226_80phon(self, freq: float) -> float:
        """ISO 226:2003 等响度曲线 80 phon，0dB 参考点在 1kHz。
        适用于混音监听场景（标准监听 SPL ~78-83 dB）。
        权重含义：负值表示该频率人耳感知较弱（需更多 SPL），与 A 计权符号约定相同。
        """
        # ISO 226:2003 Table 1 参数
        _F  = [20,25,31.5,40,50,63,80,100,125,160,200,250,315,400,500,
               630,800,1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000,12500]
        _AF = [0.532,0.506,0.480,0.455,0.432,0.409,0.387,0.367,0.349,0.330,
               0.315,0.301,0.288,0.276,0.267,0.259,0.253,0.250,0.246,0.244,
               0.243,0.243,0.243,0.242,0.242,0.245,0.254,0.271,0.301]
        _LU = [-31.6,-27.2,-23.0,-19.1,-15.9,-13.0,-10.3,-8.1,-6.2,-4.5,
               -3.1,-2.0,-1.1,-0.4,0.0,0.3,0.5,0.0,-2.7,-4.1,
               -1.0,1.7,2.5,1.2,-2.1,-7.1,-11.2,-10.7,-3.5]
        _TF = [78.5,68.7,59.5,51.1,44.0,37.5,31.5,26.5,22.1,17.9,
               14.4,11.4,8.6,6.2,4.4,3.0,2.2,2.4,3.5,1.7,
               -1.3,-4.2,-6.0,-5.4,-1.5,6.0,12.6,13.9,12.3]

        L_N = 80.0

        def _lp(af, lu, tf):
            A = 4.47e-3 * (10**(0.025 * L_N) - 1.15) + (0.4 * 10**((tf + lu) / 10 - 9))**af
            return (10 / af) * np.log10(A) - lu + 94

        lp_table = [_lp(a, l, t) for a, l, t in zip(_AF, _LU, _TF)]

        # 在对数频率轴上插值；超出 12500Hz 截断使用边界值（标准不定义此范围外的数据）
        log_freqs = np.log10(_F)
        log_f = np.log10(max(20.0, min(12500.0, freq)))
        lp = float(np.interp(log_f, log_freqs, lp_table))

        lp_1k = lp_table[17]  # 1000Hz 处的等响度 SPL = 80dB
        return lp_1k - lp     # 负值=该频率需更多 SPL→人耳不敏感；正值=更敏感

    def _get_psychoacoustic_weight(self, freq: float) -> float:
        """根据 config 的 PSYCHOACOUSTIC_MODEL 选择计权函数。"""
        model = settings.PSYCHOACOUSTIC_MODEL
        if model == "k_weighting":
            return self._get_k_weighting(freq)
        elif model == "iso226_80phon":
            return self._get_iso226_80phon(freq)
        else:
            return self._get_a_weighting(freq)

    def _calculate_masking_threshold(self, loudness_db: np.ndarray) -> np.ndarray:
        """
        [内部工具] 计算频域掩蔽阈值 (Frequency Masking Threshold)。
        响亮的声音会“掩蔽”相邻频率较弱的声音。
        """
        num_bands = len(loudness_db)
        masking_db = np.full(num_bands, -100.0) # 初始化掩蔽基底

        for i in range(num_bands):
            energy = loudness_db[i]
            if energy < -80: 
                continue # 太小声的频段没有掩蔽能力

            for j in range(num_bands):
                if i == j:
                    # 频段自身的掩蔽效应（需要高出底噪一定程度才算清晰）
                    mask = energy - 6.0 
                elif j > i:
                    # 向上掩蔽 (Upward Spread): 低频掩蔽高频，坡度较缓 (例如每频段衰减 10dB)
                    mask = energy - 10.0 * (j - i) - 6.0
                else:
                    # 向下掩蔽 (Downward Spread): 高频掩蔽低频，坡度极陡 (例如每频段衰减 18dB)
                    mask = energy - 18.0 * (i - j) - 6.0
                
                # 取各个掩蔽源对当前频段造成的最大掩蔽值
                masking_db[j] = max(masking_db[j], mask)
                
        return masking_db

    def process_and_add_snapshot(self, frames: List[List[float]], frequencies: List[float], slot_index: int):
        """处理 20 帧数据：应用心理声学模型，提取核心语义特征"""
        if not frames or not frequencies:
            return

        if slot_index < -1 or slot_index >= settings.MAX_SENSORY_SLOTS:
            print(f"⚠️ [SensoryManager] 收到非法的槽位索引: {slot_index}")
            return

        # ==========================================
        # 1. 物理数据转 Numpy (物理层)
        # ==========================================
        # frames 维度: [20, 24], 值为 0.0 ~ 1.0
        np_frames = np.array(frames)
        freqs = np.array(frequencies)
        num_bands = len(freqs)

        # 计算物理均值和峰值 (0.0 ~ 1.0)
        phys_avg = np.mean(np_frames, axis=0)
        phys_max = np.max(np_frames, axis=0)

        # 转换为分贝 (-100dB ~ 0dB) 
        phys_avg_db = phys_avg * 100.0 - 100.0

        # ==========================================
        # 2. 心理声学补偿 (感知层)
        # ==========================================
        # A. 计算并叠加 A-Weighting (等响度补偿)
        a_weights = np.array([self._get_psychoacoustic_weight(f) for f in freqs])
        perceived_loudness_db = phys_avg_db + a_weights
        
        # 将感知响度映射回 0.0 ~ 1.0 用于平衡计算
        perceived_loudness_norm = np.clip((perceived_loudness_db + 100.0) / 100.0, 0.0, 1.0)

        # B. 计算掩蔽阈值 (Masking)
        masking_threshold_db = self._calculate_masking_threshold(perceived_loudness_db)

        # ==========================================
        # 3. 语义特征提取 (推理层)
        # ==========================================
        # 整体平衡 (使用感知响度而非物理能量)
        low_energy = np.mean(perceived_loudness_norm[0:8]) if num_bands >= 8 else 0
        mid_energy = np.mean(perceived_loudness_norm[8:16]) if num_bands >= 16 else 0
        high_energy = np.mean(perceived_loudness_norm[16:24]) if num_bands == 24 else 0

        # 提取 Peaks 和 Valleys (结合掩蔽效应)
        valid_peaks = []
        valid_valleys = []
        
        for i in range(num_bands):
            f = freqs[i]
            loudness = perceived_loudness_db[i]
            masking = masking_threshold_db[i]
            
            # 判断有效 Peak：响度必须高出其掩蔽阈值至少 3dB，且物理上有一定能量
            if loudness > (masking + 3.0) and phys_avg_db[i] > -80.0:
                # 使用 perceived_loudness_norm 作为记录值传给大模型，更符合听感直觉
                valid_peaks.append({"freq": f, "energy": perceived_loudness_norm[i]})
            
            # 判断有效 Valley：没被完全掩蔽，且能量低于周边
            elif loudness < masking and phys_avg_db[i] > -90.0:
                 valid_valleys.append({"freq": f, "energy": perceived_loudness_norm[i]})

        # 排序并截取 Top N
        valid_peaks = sorted(valid_peaks, key=lambda x: x["energy"], reverse=True)[:settings.SENSORY_TOP_PEAKS]
        valid_valleys = sorted(valid_valleys, key=lambda x: x["energy"])[:settings.SENSORY_TOP_VALLEYS]

        # 动态特征 (波峰因数 Crest Factor)
        # 物理瞬态不需要 A计权，直接用 max - avg
        crest_factors = phys_max - phys_avg
        avg_crest = np.mean(crest_factors)
        
        # 细化动态标签，帮助 AI 决定是否使用 Dynamic EQ
        if avg_crest > 0.35:
            dynamic_type = "高瞬态/极具打击感 (适合使用动态EQ控制突刺)"
        elif avg_crest > 0.15:
            dynamic_type = "中等动态/弹拨乐"
        else:
            dynamic_type = "平稳/铺底/长音 (适合使用静态EQ)"

        # 记录快照
        snapshot = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "balance": {"low": low_energy, "mid": mid_energy, "high": high_energy},
            "peaks": valid_peaks,
            "valleys": valid_valleys,
            "dynamic_type": dynamic_type,
            "masked_count": num_bands - len(valid_peaks) - len(valid_valleys) # 告诉AI有多少被我们过滤了
        }
        # [修改] 快照分流存储逻辑
        if slot_index == -1:
            self.silent_snapshot = snapshot
            print(f"🤫 [SensoryManager] 接收到后台静默轨数据 (用于闭环验证)，掩蔽过滤了 {snapshot['masked_count']} 个伪频段。")
        else:
            self.snapshots[slot_index] = snapshot
            print(f"👂 [SensoryManager] 深度听觉分析完成 (掩蔽过滤了 {snapshot['masked_count']} 个伪频段)，已存入槽位 [{slot_index+1}/5]")

    def reset(self):
        """重置所有快照，用于 benchmark 每个 test case 之间的隔离"""
        self.snapshots = {}
        self.silent_snapshot = {}

    def delete_snapshot(self, slot_index: int):
        # [逻辑保持不变]
        if slot_index in self.snapshots:
            del self.snapshots[slot_index]
            print(f"🗑️ [SensoryManager] 已清除槽位 [{slot_index+1}/5] 的快照。")

    def get_silent_tracking_data(self) -> str:
        """[新增] 获取专用于闭环验证的当前实时环境状态"""
        if not self.silent_snapshot:
            return "当前没有任何实时静默轨数据。"

        snap = self.silent_snapshot
        bal = snap['balance']
        bal_str = f"低频={bal['low']:.2f}, 中频={bal['mid']:.2f}, 高频={bal['high']:.2f}"
        peaks_str = ", ".join([f"{p['freq']:.1f}Hz({p['energy']:.2f})" for p in snap["peaks"]])
        
        return f"""
[系统实时监控轨 - 调参后最新状态] ({snap['timestamp']})
1. 整体平衡: {bal_str}
2. 动态特性: {snap['dynamic_type']}
3. 当前显著共振点: {peaks_str if peaks_str else '无明显共振'}"""

    def get_sensory_data(self) -> str:
        # [逻辑保持不变]
        if not self.snapshots:
            return "当前没有用户提供的音频采样参考。"

        context_lines = ["\n[系统声学诊断] 以下音频快照已通过人类心理声学模型（A计权与掩蔽效应）校准，能量值为听感响度(0.0-1.0)："]
        
        for slot_idx in sorted(self.snapshots.keys()):
            snap = self.snapshots[slot_idx]
            bal = snap['balance']
            bal_str = f"低频={bal['low']:.2f}, 中频={bal['mid']:.2f}, 高频={bal['high']:.2f}"
            peaks_str = ", ".join([f"{p['freq']:.1f}Hz({p['energy']:.2f})" for p in snap["peaks"]])
            valleys_str = ", ".join([f"{v['freq']:.1f}Hz({v['energy']:.2f})" for v in snap["valleys"]])
            
            line = f"""
--- 槽位 #{slot_idx+1} ({snap['timestamp']}) ---
1. 整体平衡: {bal_str}
2. 动态特性: {snap['dynamic_type']}
3. 【需衰减】显著过载频率 (能量过高/刺耳轰鸣，必须用负增益压制): {peaks_str if peaks_str else '无'}
4. 【需提升】显著缺失频率 (能量不足/暗淡，必须用正增益补偿): {valleys_str if valleys_str else '无'}"""
            context_lines.append(line)
            
        return "\n".join(context_lines)

    def process_reflexion_diff(self, pre_frames: List[List[float]], post_frames: List[List[float]], frequencies: List[float]) -> str:
            """
            [新增] 处理影子录音机传来的 A/B 闭环双轨数据，生成对比报告供 LLM 反思
            """
            if not pre_frames or not post_frames or not frequencies:
                return "[系统提示：未收到有效的对比快照数据]"

            # 1. 内部辅助函数：复用之前的感知逻辑提取特征文本
            def extract_features(frames):
                np_frames = np.array(frames)
                phys_avg = np.mean(np_frames, axis=0)
                phys_avg_db = phys_avg * 100.0 - 100.0
                
                a_weights = np.array([self._get_psychoacoustic_weight(f) for f in frequencies])
                perceived_loudness_db = phys_avg_db + a_weights
                perceived_loudness_norm = np.clip((perceived_loudness_db + 100.0) / 100.0, 0.0, 1.0)
                masking_threshold_db = self._calculate_masking_threshold(perceived_loudness_db)

                # 整体平衡
                num_bands = len(frequencies)
                low_energy = np.mean(perceived_loudness_norm[0:8]) if num_bands >= 8 else 0
                mid_energy = np.mean(perceived_loudness_norm[8:16]) if num_bands >= 16 else 0
                high_energy = np.mean(perceived_loudness_norm[16:24]) if num_bands == 24 else 0
                bal_str = f"低频区={low_energy:.2f}, 中频区={mid_energy:.2f}, 高频区={high_energy:.2f}"

                # Peaks
                valid_peaks = []
                for i in range(num_bands):
                    if perceived_loudness_db[i] > (masking_threshold_db[i] + 3.0) and phys_avg_db[i] > -80.0:
                        valid_peaks.append({"freq": frequencies[i], "energy": perceived_loudness_norm[i]})
                valid_peaks = sorted(valid_peaks, key=lambda x: x["energy"], reverse=True)[:3]
                peaks_str = ", ".join([f"{p['freq']:.1f}Hz({p['energy']:.2f})" for p in valid_peaks])
                
                return bal_str, peaks_str

            # 2. 分别计算操作前和操作后的特征
            pre_bal, pre_peaks = extract_features(pre_frames)
            post_bal, post_peaks = extract_features(post_frames)

            # 3. 组装给大模型的 Prompt 报告
            diff_report = f"""
[系统声学环境拦截器 - 闭环验证报告]
你刚刚执行了 EQ 调整。这是操作前后的物理听觉特征对比：

【操作前 (问题状态)】
- 整体平衡: {pre_bal}
- 显著共振点: {pre_peaks if pre_peaks else '无明显突刺'}

【操作后 (当前最新状态)】
- 整体平衡: {post_bal}
- 显著共振点: {post_peaks if post_peaks else '无明显突刺'}

【注意】
1. 区分“瑕疵”与“预期效果”：如果【操作后】新出现的显著共振点，恰好是你刚刚主动提升的频段（比如你为了增加亮度提升了高频），这是正常的预期效果！绝对不要把它当成“刺耳瑕疵”又降回去！
2. 只有当出现了你没有去动，但因为系统联动导致的新频段严重淤积时，才需要进行二段微调。
3. 如果当前状态已经符合你刚才的调整预期，请立即停止调用工具，直接向用户汇报验证成功！
"""
            print("🤫 [SensoryManager] 成功生成影子 A/B 对比报告！")
            return diff_report