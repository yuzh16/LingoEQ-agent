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

    def _detect_instrument(self, text: str) -> list:
        """
        轻量级意图路由：根据用户输入提取乐器实体。
        自动兼容通用规则 (general) 和非贝斯规则 (non_bass)。
        【架构升级】：支持多乐器并发检测 (Multi-Intent Detection)
        """
        text_lower = text.lower()
        # 默认始终包含通用混音法则
        target = ["general"]
        
        # 智能兼容：只要用户没提贝斯，就允许触发 "non_bass" (低频清理) 规则
        if not any(word in text_lower for word in ["贝斯", "bass", "808"]):
            target.append("non_bass")
            
        # 具体的乐器实体探测 (改用独立 if，支持一句话包含多个乐器)
        if any(word in text_lower for word in ["人声", "唱", "vocal", "男声", "女声", "齿音", "鼻音"]):
            target.append("vocal")
            
        if any(word in text_lower for word in ["鼓", "kick", "drum", "底鼓", "军鼓", "打击乐"]):
            target.append("drum")
            
        if any(word in text_lower for word in ["吉他", "guitar", "木吉他", "电吉他"]):
            target.append("guitar")
            
        if any(word in text_lower for word in ["贝斯", "bass"]):
            target.append("bass")
            
        if any(word in text_lower for word in ["钢琴", "piano", "键盘", "keyboard", "电钢"]):
            target.append("piano")
            
        if any(word in text_lower for word in ["弦乐", "string", "提琴", "小提琴", "大提琴", "cello", "violin"]):
            target.append("string")
            
        return target


    def search(self, query: str, n_results: int = 3, distance_threshold: float = 1) -> str:
        """
        根据用户查询检索相关知识
        :param distance_threshold: L2 距离阈值。默认 1。
                                  越小要求越严格 (0表示完全一样)，大于此距离的将被过滤掉！
        :return: 拼接好的上下文文本
        """
        # 1. 将用户问题向量化
        query_embedding = self.embed_model.encode([query]).tolist()
        
        # 2. 【新增架构】意图路由：提取用户想要处理的乐器分类
        target_instruments = self._detect_instrument(query)
        print(f"🎛️ [RAG Router] 锁定目标乐器分类: {target_instruments}")
        
        # 3. 在数据库中搜索最近邻 (增加 Metadata Routing 过滤)
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where={"instrument": {"$in": target_instruments}}, # 👈 【新增】：只在匹配的分类中搜索！
            include=["documents", "distances", "metadatas"] # 👈 把 metadatas 也拿出来
        )
        
        if not results['documents'] or not results['documents'][0]:
            return ""
            
        retrieved_docs = results['documents'][0]
        distances = results['distances'][0]
        retrieved_metas = results['metadatas'][0] # 提取元数据列表
        
        valid_contexts = [] # 用于存放组装好的结构化上下文
        
        # 4. 核心逻辑：阈值过滤 (Semantic Thresholding)
        print(f"🔬 [RAG 诊断] 分析查询: '{query}'")
        for doc, dist, meta in zip(retrieved_docs, distances, retrieved_metas):
            # 距离小于等于阈值，说明足够相似，保留
            if dist <= distance_threshold:
                # 将触发的元数据提取出来，转换为绝对精准的操作指导
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
                
                # 保留你原本优秀的诊断打印逻辑，稍微截断一下长文档
                print(f"  ✅ [命中] 距离: {dist:.3f} | 匹配意图: {doc[:30]}...")
            # 距离大于阈值，说明是在“强行凑数”，抛弃
            else:
                print(f"  🛑 [拦截] 距离: {dist:.3f} | 匹配意图: {doc[:30]}... (无关内容已舍弃)")
        
        # 5. 如果所有结果都被拦截了，说明用户只说了“开启动态”这类无实质混音语义的话
        if not valid_contexts:
            print("🚫 [RAG] 所有检索结果均低于阈值，已彻底阻断知识库对 LLM 的干扰。")
            return ""
            
        # 6. 格式化返回结果
        # 返回的是拥有高度约束力的结构化上下文，极大降低 LLM 幻觉
        context_str = "### 领域专家知识指导 ###\n" + "\n---\n".join(valid_contexts) + "\n#########################"
        return context_str