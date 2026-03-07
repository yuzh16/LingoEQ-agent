# app/services/memory_manager.py
import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from app.services.state_manager import StateManager

class MemoryManager:
    def __init__(self, persist_directory: str = "data/vector_db"):
        print(f"[Memory] 正在初始化预设与记忆管理器...")
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        # 复用你的向量数据库路径
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # 专属的预设库集合，配置L2空间
        self.preset_collection = self.chroma_client.get_or_create_collection(
            name="user_presets",
            metadata={"hnsw:space": "l2"}
        )

    def save_preset(self, preset_name: str, state_manager: StateManager) -> str:
        """
        [优化版] 稀疏保存：只存储有效频段，大幅压缩数据量
        """
        effective_bands = []
        
        for idx, band in state_manager.bands.items():
            # 定义：什么是“有效”的频段？
            # 1. 必须是开启状态
            if not band.enabled:
                continue
                
            # 2. 判断是否有实质性影响
            is_cut_filter = band.filter_type in ["LowCut", "HighCut"]
            has_gain = abs(band.gain) > 0.05 # 给一点点容差
            is_dynamic = band.dynamic_enabled
            
            # 如果是切除类，或者有增益，或者有动态，才算有效
            if is_cut_filter or has_gain or is_dynamic:
                # 导出并注入索引 (保留我们刚才修复的逻辑)
                band_data = band.model_dump(by_alias=True)
                band_data["band_index"] = idx
                effective_bands.append(band_data)

        # 如果一个有效频段都没有（用户存了个寂寞），至少存一个空的 list
        if not effective_bands:
            print(f"⚠️ [Memory] 用户试图保存一个空预设: {preset_name}")

        preset_id = f"preset_{hash(preset_name)}"
        embedding = self.embed_model.encode([preset_name]).tolist()

        self.preset_collection.upsert(
            documents=[preset_name],
            embeddings=embedding,
            metadatas=[{"eq_state": json.dumps(effective_bands, ensure_ascii=False)}],
            ids=[preset_id]
        )
        
        count = len(effective_bands)
        print(f"💾 [Memory] 稀疏预设【{preset_name}】已保存 (仅包含 {count}/24 个有效频段)。")
        return f"好的！预设【{preset_name}】已保存，这其中包括 {count} 个有效调整点。"

    def load_preset(self, query_name: str, distance_threshold: float = 1.25):
        """语义检索并返回 24 频段的 JSON 数组"""
        embedding = self.embed_model.encode([query_name]).tolist()
        
        results = self.preset_collection.query(
            query_embeddings=embedding,
            n_results=1
        )
        
        if not results['documents'] or not results['documents'][0]:
            return None, "抱歉，我的记忆库里目前没有任何预设。"
            
        dist = results['distances'][0][0]
        matched_name = results['documents'][0][0]
        
        # 使用 L2 距离判断用户说的预设和库里的预设语义是否接近
        if dist <= distance_threshold:
            state_str = results['metadatas'][0][0]["eq_state"]
            print(f"📂 [Memory] 成功匹配预设【{matched_name}】(距离: {dist:.3f})")
            return json.loads(state_str), f"没问题！已经为您加载预设：【{matched_name}】。"
        else:
            print(f"🛑 [Memory] 预设匹配失败，距离太远: {dist:.3f}")
            return None, f"抱歉，我没有找到叫“{query_name}”的预设，最接近的是“{matched_name}”，但差别有点大。"

    def get_all_preset_names(self) -> str:
        """获取所有预设名称，供大模型参考"""
        try:
            results = self.preset_collection.get(include=["documents"])
            docs = results.get("documents", [])
            if not docs:
                return "暂无预设"
            return ", ".join(docs)
        except Exception:
            return "暂无预设"
        