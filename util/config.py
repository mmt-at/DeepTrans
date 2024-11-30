import os
import dotenv
dotenv.load_dotenv()
PPLX_API_KEY = os.getenv("PPLX_API_KEY")
QWEN_API_KEY=os.getenv("QWEN_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY")
OLLAMA_API_KEY=os.getenv("OLLAMA_API_KEY")
from enum import Enum, auto

class Language(Enum):
    C = "c"
    CPP = "c++"
    PYTHON = "python"
    JAVA = "java"
    CUDA = "cuda"

class PPLXModel(str, Enum):
    MIXTRAL = "mixtral-8x7b-instruct"
    LLAMA_SONAR_LARGE = "llama-3.1-sonar-large-128k-chat"
    LLAMA_SONAR_HUGE = "llama-3.1-sonar-huge-128k-online"
    LLAMA_70B = "llama-3.1-70b-instruct"

class GPTModel(str, Enum):
    GPT35_TURBO = "gpt-3.5-turbo"
    GPT4_TURBO = "gpt-4-turbo"
    GPT4 = "gpt-4"
    GPT4O_MINI = "gpt-4o-mini"
    GPT4O = "gpt-4o"
    O1_MINI = "o1-mini"
    O1_PREVIEW = "o1-preview"

class AnthropicModel(str, Enum):
    CLAUDE_35_SONNET = "claude-3-5-sonnet-latest"
    CLAUDE_35_HAIKU = "claude-3-5-haiku-latest"
    CLAUDE_3_OPUS = "claude-3-opus-latest"
    CLAUDE_3_SONNET = "claude-3-sonnet-latest"
    CLAUDE_3_HAIKU = "claude-3-haiku-latest"

class DeepseekModel(str, Enum):
    CODER = "deepseek-coder"

class QwenModel(str, Enum):
    MAX_0919 = "qwen-max-0919"
    MAX = "qwen-max"
    CODER_TURBO_LATEST = "qwen-coder-turbo-latest"
    LONG = "qwen-long"
    CODER_PLUS = "qwen-coder-plus"
    CODER_32B = "qwen2.5-coder-32b-instruct"
    CODER_TURBO = "qwen-coder-turbo"
    CODER_7B = "qwen2.5-coder-7b-instruct"

class OllamaModel(str, Enum):
    LLAMA31 = "llama3.1"
    CODEGEEX4 = "codegeex4"
    CODEGEEX4_9B = "codegeex4-9b-all-fp16"
    CODESTRAL = "codestral"
    CODESTRAL_22B = "codestral:22b-v0.1-f16"
    DEEPSEEK_V2 = "deepseek-coder-v2"
    DEEPSEEK_V2_16B = "deepseek-coder-v2:16b-lite-instruct-fp16"
    CODEGEMMA = "codegemma"
    CODEQWEN = "codeqwen"

# 如果需要获取所有可用模型列表，可以使用以下方式：
PPLX_AVAILABLE_MODELS = [model.value for model in PPLXModel]
GPT_AVAILABLE_MODELS = [model.value for model in GPTModel]
ANTHROPIC_AVAILABLE_MODELS = [model.value for model in AnthropicModel]
DEEPSEEK_AVAILABLE_MODELS = [model.value for model in DeepseekModel]
QWEN_AVAILABLE_MODELS = [model.value for model in QwenModel]
OLLAMA_AVAILABLE_MODELS = [model.value for model in OllamaModel]

import logging
logger = logging.getLogger("shared_logger")
logger.setLevel(logging.DEBUG)

import os
cur_folder = os.path.dirname(os.path.abspath(__file__))
log_folder = os.path.join(cur_folder, "../logs")
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
# 创建一个文件处理器
file_handler = logging.FileHandler(os.path.join(log_folder, "deeptrans.log"), encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

# 创建一个日志格式器并将其添加到处理器
formatter = logging.Formatter("%(asctime)s - %(threadName)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# 将处理器添加到 logger
logger.addHandler(file_handler)

logger.info("Logger initialized")
# logger.info(f"VALID MODELS:\n{PPLX_AVAILABLE_MODELS}\n{GPT_AVAILABLE_MODELS}\n{ANTHROPIC_AVAILABLE_MODELS}\n{DEEPSEEK_AVAILABLE_MODELS}\n{QWEN_AVAILABLE_MODELS}\n{OLLAMA_AVAILABLE_MODELS}")