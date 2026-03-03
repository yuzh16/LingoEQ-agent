# mock_server.py (最终修正版)
import asyncio
import json

HOST = '127.0.0.1'
PORT = 8888

def wrap_command(tool_name, args):
    """
    将简单的指令包装成 AgentClient 期待的复杂协议格式 (信封)
    """
    return {
        "result": {
            "content": [
                {
                    "type": "tool_use",       # AgentClient 识别标志
                    "tool": tool_name,        # PluginProcessor 识别标志
                    "args": args              # 参数载荷
                }
            ]
        }
    }

async def handle_client(reader, writer):
    addr = writer.get_extra_info('peername')
    print(f"🎉 C++ 插件已连接! (来自 {addr})")

    print("⏳ 等待 2 秒，让插件准备好...")
    await asyncio.sleep(2)

    # --- 指令 1: 提亮中频 ---
    print(">> 发送指令 1: Band 5, 1000Hz, +12dB")
    
    # 1. 准备数据
    cmd1_args = {
        "band_index": 5,
        "freq": 1000.0,
        "gain": 12.0,
        "q": 1.0,
        "enabled": True
    }
    # 2. 包装信封
    packet1 = wrap_command("adjust_eq_band", cmd1_args)
    
    # 3. 序列化 + 核心修复 (加上 \n)
    msg1 = json.dumps(packet1).encode('utf-8') + b'\n'
    
    # 4. 发送
    writer.write(msg1)
    await writer.drain()
    print("✅ 指令 1 已发送 (含换行符与完整信封)")

    await asyncio.sleep(2)

    # --- 指令 2: 切除低频 ---
    print(">> 发送指令 2: Band 0, 50Hz, -24dB")
    cmd2_args = {
        "band_index": 0,
        "freq": 50.0,
        "gain": -24.0,
        "q": 0.7,
        "enabled": True
    }
    packet2 = wrap_command("adjust_eq_band", cmd2_args)
    msg2 = json.dumps(packet2).encode('utf-8') + b'\n'
    
    writer.write(msg2)
    await writer.drain()
    print("✅ 指令 2 已发送")

    # 保持连接一会儿，防止数据没发完就断开
    await asyncio.sleep(1)
    print("👋 测试结束")
    writer.close()

async def main():
    server = await asyncio.start_server(handle_client, HOST, PORT)
    print(f"🚀 Mock Server 正在监听 {HOST}:{PORT}...")
    print("👉 请重启 Cubase 或重新加载 LingoEQ 插件以触发连接...")
    
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
