import os
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
load_dotenv()

class Settings:
    """
    LingoEQ 系统全局配置中心 (Global Configuration Center)
    统一管理 API 密钥、系统超参数、物理护栏阈值与网络通信设置
    """
    
    # ==========================================
    # 1. 大语言模型与核心计算层 (LLM & Inference)
    # ==========================================
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")           # 填你所用 provider 的 API Key
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "deepseek-ai/DeepSeek-V3")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))   # 推理确定性，越低越稳定
    MAX_REFLEXION_DEPTH = int(os.getenv("MAX_REFLEXION_DEPTH", "2"))  # 闭环反思最大深度

    # ==========================================
    # 2. 网络通信与系统调度 (Network & IPC)
    # ==========================================
    # 用于 Python 后端与 C++ (JUCE) 宿主插件的 TCP/WebSocket 通信
    WS_HOST = os.getenv("WS_HOST", "127.0.0.1")
    WS_PORT = int(os.getenv("WS_PORT", "8888"))           # 必须与 C++ 端的通信端口严格对齐

    # ==========================================
    # 3. 本地存储与初始数据 (Storage & Initialization)
    # ==========================================
    VECTOR_DB_DIR = "data/vector_db"                 # ChromaDB 向量持久化目录
    FACTORY_PRESETS_DIR = "data/factory_presets"     # 出厂预设文件夹路径
    INITIAL_KNOWLEDGE_BASE = "data/knowledge_base/mixing_tips.json" # 专家混音法则初始 JSON

    # ==========================================
    # 4. RAG 与语义记忆引擎 (Knowledge & Memory)
    # ==========================================
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"        # 文本向量化模型
    RAG_COLLECTION_NAME = "mixing_knowledge_v2"      # RAG 知识集合空间名
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))                    # 每次向大模型注入的专家知识条数
    RAG_DISTANCE_THRESHOLD = float(os.getenv("RAG_DISTANCE_THRESHOLD", "1.125"))   # RAG 检索 L2 距离阈值
    PRESET_DISTANCE_THRESHOLD = float(os.getenv("PRESET_DISTANCE_THRESHOLD", "1.3"))  # 预设匹配 L2 距离阈值
    MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "5"))     # 携带的最大历史对话轮数

    # ==========================================
    # 5. 心理声学与感知特征 (Psychoacoustics & Sensory)
    # ==========================================
    MAX_SENSORY_SLOTS = int(os.getenv("MAX_SENSORY_SLOTS", "5"))    # 允许缓存的参考音频快照数量上限
    SENSORY_TOP_PEAKS = int(os.getenv("SENSORY_TOP_PEAKS", "3"))    # 每次汇报的最显著共振刺耳点数量
    SENSORY_TOP_VALLEYS = int(os.getenv("SENSORY_TOP_VALLEYS", "2")) # 每次汇报的最显著频率凹陷点数量

    # ==========================================
    # 6. 确定性物理护栏 (Deterministic Guardrails)
    # ==========================================
    SAFETY_MARGIN = float(os.getenv("SAFETY_MARGIN", "1.2"))        # DSP 频段干涉计算的安全缓冲系数

    # 绝对声学安全钳位极限 (Safety Clamping Limits)
    # 严防 LLM 幻觉产生烧毁音箱或导致宿主崩溃的非法参数
    GAIN_LIMIT_MIN = float(os.getenv("GAIN_LIMIT_MIN", "-24.0"))
    GAIN_LIMIT_MAX = float(os.getenv("GAIN_LIMIT_MAX", "18.0"))
    FREQ_LIMIT_MIN = float(os.getenv("FREQ_LIMIT_MIN", "20.0"))
    FREQ_LIMIT_MAX = float(os.getenv("FREQ_LIMIT_MAX", "20000.0"))
    Q_LIMIT_MIN = float(os.getenv("Q_LIMIT_MIN", "0.1"))
    Q_LIMIT_MAX = float(os.getenv("Q_LIMIT_MAX", "10.0"))

# 实例化单例，全局只使用这一个 settings 对象
settings = Settings()
