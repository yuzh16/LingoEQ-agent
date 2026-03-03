import socket
import json
import time

def test_agent_interaction():
    HOST = '127.0.0.1'
    PORT = 8000

    # 模拟几种不同的用户输入，测试 Agent 的不同反应
    test_queries = [
        # 测试 1: 纯聊天
        "你好，你是谁？你能做什么？",
        
        # 测试 2: 复杂的混音需求 (触发 RAG + Tool + Thought)
        "我想要让人声听起来更温暖，去除那种很薄的感觉。"
    ]

    print(f"🔌 [Client] 正在连接 {HOST}:{PORT} ...")

    for query in test_queries:
        print("\n" + "-"*60)
        print(f"🗣️ [User] 发送指令: {query}")
        
        request_data = {
            "jsonrpc": "2.0",
            "method": "chat/send", 
            "params": {"content": query},
            "id": int(time.time())
        }
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST, PORT))
                
                # 发送 (加换行符)
                msg_str = json.dumps(request_data) + "\n"
                s.sendall(msg_str.encode('utf-8'))

                # 接收
                print("⏳ [Client] 等待 Agent 思考...")
                received = s.recv(8192) # 缓冲区加大，因为包含了思维链，文本量可能较大
                
                if not received:
                    print("❌ [Client] 未收到数据")
                    continue

                response_str = received.decode('utf-8')
                response_json = json.loads(response_str)

                # --- 结果解析与可视化 ---
                if "result" in response_json:
                    content_list = response_json["result"]["content"]
                    
                    print(f"✨ [Agent 响应]:")
                    
                    for item in content_list:
                        # 1. 展示思维链 (用灰色或特殊标记)
                        if item["type"] == "text" and item.get("role") == "thought":
                            print(f"\n🧠 [思维链]:\n{item['text']}")
                        
                        # 2. 展示正式回复
                        elif item["type"] == "text" and item.get("role") == "assistant":
                            print(f"\n💬 [回复]:\n{item['text']}")
                            
                        # 3. 展示工具调用
                        elif item["type"] == "tool_use":
                            print(f"\n🛠️ [调用工具] -> {item['name']}")
                            print(f"   参数: {json.dumps(item['input'], indent=2, ensure_ascii=False)}")
                            
                elif "error" in response_json:
                    print(f"❌ Agent 报错: {response_json['error']}")

        except Exception as e:
            print(f"❌ 发生错误: {e}")
            break
        
        # 休息一下，模拟真实交互间隔
        time.sleep(1)

if __name__ == "__main__":
    test_agent_interaction()
