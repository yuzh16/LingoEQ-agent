import os
import json
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import glob
import chromadb
from sentence_transformers import SentenceTransformer
from app.services.state_manager import StateManager

class MemoryManager:
    def __init__(self, persist_directory: str = "data/vector_db", factory_dir: str = "data/factory_presets"):
        print(f"[Memory] 正在初始化预设与记忆管理器...")
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # 统一预设库集合
        self.preset_collection = self.chroma_client.get_or_create_collection(
            name="user_presets",
            metadata={"hnsw:space": "l2"}
        )

        # 🚀 新增：启动时加载出厂预设
        self.factory_dir = factory_dir
        self._ingest_factory_presets()

    def _ingest_factory_presets(self):
        """
        扫描 factory_dir 下的 JSON 文件并存入向量库
        """
        if not os.path.exists(self.factory_dir):
            print(f"⚠️ [Memory] 出厂预设目录不存在: {self.factory_dir}")
            return

        json_files = glob.glob(os.path.join(self.factory_dir, "*.json"))
        print(f"[Memory] 发现 {len(json_files)} 个出厂预设文件，正在同步...")

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for filepath in json_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 1. 构造唯一 ID (factory_文件名)
                filename = os.path.basename(filepath).replace(".json", "")
                preset_id = f"factory_{filename}"
                
                # 2. 构造搜索文本 (Name + Description + Tags)
                # 这样用户搜 "Bright Vocal" 时能命中 Tag
                search_text = f"{data['name']} {data.get('description', '')} {' '.join(data.get('tags', []))}"
                
                # 3. 构造 Metadata (包含实际参数)
                # 注意：这里我们只存 bands 列表，与 save_preset 保持结构一致
                eq_state_json = json.dumps(data["bands"], ensure_ascii=False)
                
                meta = {
                    "eq_state": eq_state_json,
                    "is_factory": True,       # 标记为出厂预设
                    "category": data.get("category", "General"),
                    "description": data.get("description", ""),
                    "name": data["name"]
                }

                ids.append(preset_id)
                documents.append(search_text)
                metadatas.append(meta)

            except Exception as e:
                print(f"❌ [Memory] 加载预设 {filepath} 失败: {e}")

        # 批量入库 (Upsert: 有则更新，无则插入)
        if ids:
            embeddings = self.embed_model.encode(documents).tolist()
            self.preset_collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            print(f"[Memory] ✅ 已同步 {len(ids)} 个出厂预设到向量库。")

    def save_preset(self, preset_name: str, state_manager: StateManager) -> str:
        """
        [保持不变] 稀疏保存用户预设
        """
        # ... (这里完全复用之前的逻辑) ...
        effective_bands = []
        for idx, band in state_manager.bands.items():
            if not band.enabled: continue
            is_cut_filter = band.filter_type in ["LowCut", "HighCut"]
            has_gain = abs(band.gain) > 0.05 
            is_dynamic = band.dynamic_enabled
            
            if is_cut_filter or has_gain or is_dynamic:
                band_data = band.model_dump(by_alias=True)
                band_data["band_index"] = idx
                effective_bands.append(band_data)

        if not effective_bands:
            print(f"⚠️ [Memory] 用户试图保存一个空预设: {preset_name}")

        preset_id = f"user_{hash(preset_name)}" # 区分 user_ 前缀
        embedding = self.embed_model.encode([preset_name]).tolist()

        self.preset_collection.upsert(
            documents=[preset_name],
            embeddings=embedding,
            metadatas=[{
                "eq_state": json.dumps(effective_bands, ensure_ascii=False),
                "is_factory": False, # 标记为用户预设
                "category": "User",
                "description": "User created preset"
            }],
            ids=[preset_id]
        )
        
        count = len(effective_bands)
        print(f"💾 [Memory] 用户预设【{preset_name}】已保存。")
        return f"好的！预设【{preset_name}】已保存。"

    def load_preset(self, query_name: str, distance_threshold: float = 1.3): # 稍微放宽一点阈值
        """语义检索并返回 JSON"""
        embedding = self.embed_model.encode([query_name]).tolist()
        
        results = self.preset_collection.query(
            query_embeddings=embedding,
            n_results=1
        )
        
        if not results['documents'] or not results['documents'][0]:
            return None, "抱歉，我的记忆库里目前没有任何预设。"
            
        dist = results['distances'][0][0]
        metadata = results['metadatas'][0][0]
        
        is_factory = metadata.get("is_factory", False)
        source_str = "【官方推荐】" if is_factory else "【用户收藏】"

        if dist <= distance_threshold:
            state_str = metadata["eq_state"]
            desc = metadata.get("description", "")
            
            # 👇【新增】从 metadata 里拿出刚才存的漂亮名字
            # 如果是老数据没有 name 字段，就用 "未命名" 兜底
            real_name = metadata.get("name", "未命名预设")
            
            print(f"📂 [Memory] 匹配到 {real_name} (距离: {dist:.3f})")
            
            # 👇【修改】返回的信息里带上 real_name
            return json.loads(state_str), f"已为您加载{source_str}：**{real_name}**。\n({desc})"
        else:
            print(f"🛑 [Memory] 匹配失败，距离: {dist:.3f}")
            return None, f"未找到类似“{query_name}”的预设。"

    def get_all_preset_names(self) -> str:
        """
        [关键修改] 获取所有预设名称，并按分类整理，供 ContextManager 使用
        """
        try:
            # 获取所有数据
            results = self.preset_collection.get(include=["metadatas", "documents"])
            metas = results.get("metadatas", [])
            docs = results.get("documents", [])
            
            if not metas:
                return "暂无预设"
            
            # 分类整理
            factory_list = []
            user_list = []
            
            for doc, meta in zip(docs, metas):
                # 简单处理：取 document 的前 20 个字符作为名字展示
                name = doc.split(" ", 1)[0] 
                if meta.get("is_factory"):
                    factory_list.append(f"{name} ({meta.get('category')})")
                else:
                    user_list.append(name)
            
            output = []
            if factory_list:
                output.append(f"🏭 [官方推荐]: {', '.join(factory_list)}")
            if user_list:
                output.append(f"👤 [用户收藏]: {', '.join(user_list)}")
                
            return "\n".join(output)

        except Exception as e:
            print(f"❌ [Memory] 获取预设列表失败: {e}")
            return "暂无预设"
