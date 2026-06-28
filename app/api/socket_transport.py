import asyncio
import json
from app.dispatcher.server import AgentDispatcher
from config import settings

class SocketServer:
    def __init__(self, host=settings.WS_HOST, port=settings.WS_PORT, rag_engine=None):
        self.host = host
        self.port = port
        self.dispatcher = AgentDispatcher(rag_engine=rag_engine)

    async def handle_client(self, reader, writer):
        addr = writer.get_extra_info('peername')
        print(f"🔌 [Socket] C++ 客户端已连接: {addr}")

        try:
            while True:
                # 读取一行 (以 \n 结尾)
                # 这是一个异步等待操作，不会阻塞 Loop
                data = await reader.readuntil(b'\n')
                message = data.decode().strip()
                
                if not message:
                    continue

                # 解析 JSON
                try:
                    request_json = json.loads(message)
                    
                    # -------------------------------------------------------
                    # 【核心修改点】
                    # 使用 asyncio.to_thread 将同步的 MCP 处理逻辑扔到线程池运行
                    # 这样 RAG 检索和 LLM 请求的几秒钟等待期，不会卡死整个 Socket 服务
                    # -------------------------------------------------------
                    print(f"🔄 [Socket] 正在处理请求 (Offloading to thread)...")
                    response_data = await asyncio.to_thread(
                        self.dispatcher.process_request,
                        request_json
                    )
                    
                    # 发回响应 (加上 \n)
                    response_str = json.dumps(response_data) + "\n"
                    writer.write(response_str.encode())
                    await writer.drain()
                    
                    print(f"📤 [Socket] 已发送回复给 C++")

                except json.JSONDecodeError:
                    print("⚠️ [Socket] JSON 解析错误")
                    # 这里也可以选择发回一个错误响应给客户端

        except asyncio.IncompleteReadError:
            print(f"🔌 [Socket] 客户端断开连接")
        except Exception as e:
            print(f"⚠️ [Socket Error] {e}")
        finally:
            writer.close()

    async def start(self):
        server = await asyncio.start_server(
            self.handle_client, self.host, self.port
        )
        print(f"🚀 LingoEQ Agent 正在监听 {self.host}:{self.port} ...")
        async with server:
            await server.serve_forever()
