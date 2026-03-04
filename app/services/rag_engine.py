import os
import json # 【新增】导入 json 模块处理结构化数据
# 【新增】强制使用国内镜像站，解决连接超时问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import chromadb
from typing import List
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

class RagEngine:
    def __init__(self, persist_directory: str = "data/vector_db"):
        """
        初始化 RAG 引擎
        :param persist_directory: 向量数据库的存储路径
        """
        print(f"[RAG] 正在初始化 RAG 引擎，存储路径: {persist_directory}...")
        
        # 1. 初始化 Embedding 模型 (使用轻量级的 all-MiniLM-L6-v2)
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. 初始化 ChromaDB 客户端
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # 3. 获取或创建集合 (Collection)
        # 【架构修改】更名为 mixing_knowledge_v2。既实现了版本隔离，又能在下面配合清空逻辑
        self.collection_name = "mixing_knowledge_v2"
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "l2"} # 【新增】显式声明 L2 距离，确保你的阈值逻辑精准运行
        )
        
        print("[RAG] 引擎初始化完成。")

    def ingest_knowledge(self, file_path: str, force_reload: bool = False):
        """
        从结构化 JSON 文件读取知识并存入向量数据库 (具备缓存与秒级启动能力)
        :param file_path: JSON 知识库路径
        :param force_reload: 是否强制清空并重新计算向量 (默认 False)
        """
        if not os.path.exists(file_path):
            print(f"[RAG Error] 知识库文件不存在: {file_path}")
            return

        # 1. 读取当前的 JSON 数据
        with open(file_path, 'r', encoding='utf-8') as f:
            knowledge_data = json.load(f)

        if not knowledge_data:
            print("[RAG] 文件为空或JSON无数据，跳过入库。")
            return

        json_item_count = len(knowledge_data)
        db_item_count = self.collection.count()

        # 2. 【核心架构点：秒级启动拦截】
        # 如果不是强制重载，且数据库里的数据量和 JSON 里的数量一致，直接跳过计算！
        if not force_reload and db_item_count == json_item_count:
            print(f"[RAG ⚡] 检测到本地已存在 {db_item_count} 条知识向量，且与文件数量一致，跳过 Embedding，实现秒级启动！")
            return

        print(f"[RAG] 正在重载结构化知识库... (数据库现有: {db_item_count}条, JSON包含: {json_item_count}条)")

        # 3. 执行强制清空与重载逻辑
        try:
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name, 
                metadata={"hnsw:space": "l2"}
            )
            print("[RAG] 已清空旧向量空间，准备写入全新结构化数据。")
        except ValueError:
            pass # 如果集合本来就不存在则忽略

        ids = []
        documents = [] 
        metadatas = [] 
        
        for item in knowledge_data:
            ids.append(item["id"])
            search_text = f"意图: {', '.join(item['intent'])}. 听感: {item['acoustic_description']}"
            documents.append(search_text)
            
            metadatas.append({
                "instrument": item["target_instrument"],
                "action_payload": json.dumps(item["action"], ensure_ascii=False),
                "reasoning_hint": item["reasoning_hint"]
            })

        print(f"[RAG] 正在调用模型生成 {len(documents)} 条数据的向量，这可能需要几秒钟...")
        embeddings = self.embed_model.encode(documents).tolist()

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas, 
            ids=ids
        )
        print(f"[RAG ✅] 成功入库并持久化 {len(documents)} 条结构化知识！")

    def search(self, intent_data: dict, n_results: int = 4, distance_threshold: float = 1.25) -> str:
        """
        [Agentic RAG 核心检索]：接收 Router 分析出的结构化数据，进行提纯定向检索
        :param intent_data: 包含 'instruments' 列表和 'keywords' 列表的字典
        :param distance_threshold: L2 距离阈值。放宽到 1.25 以适应提纯后的高密度特征词
        :return: 拼接好的强约束力 Context 文本
        """
        # 1. 解析 LLM 给出的字典
        instruments = intent_data.get("instruments", ["general"])
        # 始终混入 general，确保通用混音法则永远能被搜到 (平替了原先的硬编码兼容逻辑)
        if "general" not in instruments:
            instruments.append("general")
            
        keywords = intent_data.get("keywords", [])
        # 兜底：如果 Router 没提取出 keywords，把 instruments 当作关键词
        if not keywords:
             keywords = instruments
             
        # 2. 【核心架构点：高密度文本拼接】将提取出的精确特征词组合，去除废话
        query_text = " ".join(keywords)
        print(f"🔍 [RagEngine] Agentic 提纯词组: '{query_text}' -> 搜索范围 (硬过滤): {instruments}")
        
        # 3. 向量化提纯后的关键词组
        query_embedding = self.embed_model.encode([query_text]).tolist()
        
        # 4. 执行绝对分类过滤与 L2 检索
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where={"instrument": {"$in": instruments}}, # 👈 仅在 Router 指定的乐器分类中检索
            include=["documents", "distances", "metadatas"]
        )
        
        if not results['documents'] or not results['documents'][0]:
            return ""
            
        retrieved_docs = results['documents'][0]
        distances = results['distances'][0]
        retrieved_metas = results['metadatas'][0] 
        
        valid_contexts = [] 
        
        print(f"🔬 [RAG 诊断] 开始匹配空间距离...")
        # 5. 核心逻辑：阈值过滤 (Semantic Thresholding)
        for doc, dist, meta in zip(retrieved_docs, distances, retrieved_metas):
            if dist <= distance_threshold:
                action_json = meta["action_payload"]
                hint = meta["reasoning_hint"]
                instrument = meta["instrument"]
                
                # 组装给 LLM 的终极 Context，彻底锁死 JSON 格式
                structured_context = (
                    f"  [目标乐器: {instrument}]\n"
                    f"  分析依据: {hint}\n"
                    f"  建议操作参数 (请严格参考此 JSON 范围生成工具调用): {action_json}"
                )
                valid_contexts.append(structured_context)
                print(f"  ✅ [命中] 距离: {dist:.3f} | 匹配依据: {hint}")
            else:
                print(f"  🛑 [拦截] 距离: {dist:.3f} | 匹配意图过远已舍弃")

        if not valid_contexts:
            print("🚫 [RAG] 未找到低于阈值的匹配项，已彻底阻断知识库对 LLM 的干扰。")
            return ""
            
        # 6. 格式化返回结果，形成对 Action LLM 的绝对指导
        return "### 领域专家知识指导 ###\n" + "\n---\n".join(valid_contexts) + "\n#########################"