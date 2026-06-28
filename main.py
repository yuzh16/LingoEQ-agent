import os
os.environ["HF_HUB_OFFLINE"] = "1"

import asyncio
import colorama
from config import settings
from app.api.socket_transport import SocketServer
from app.services.rag_engine import RagEngine # [新增] 用于预加载数据

# 初始化颜色输出
colorama.init(autoreset=True)

def initialize_system():
    print("🚀 [System] 正在初始化 LingoEQ 系统组件...")
    try:
        rag = RagEngine()
        rag.ingest_knowledge(settings.INITIAL_KNOWLEDGE_BASE) 
        return rag # 【关键】：必须返回这个已经装载知识的实例
    except Exception as e:
        print(f"❌ [Error] RAG 引擎初始化失败: {e}")
        return None

if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # 1. 拿到唯一且聪明的 RAG 实例
    shared_rag = initialize_system()
    
    # 2. 启动 Socket 服务器，并把这个实例“注入”进去
    # 这样整个系统就共用同一个模型、同一个向量库缓存
    server = SocketServer(rag_engine=shared_rag) 
    
    try:
        # 注意：此处端口应与 C++ 端配置一致 (8888)
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\n🛑 服务器已停止") 