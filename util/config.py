import os
PPLX_API_KEY = os.getenv("PPLX_API_KEY")
QWEN_API_KEY=os.getenv("QWEN_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY")
PPLX_AVAILABLE_MODELS = [
    "mixtral-8x7b-instruct",
    "llama-3.1-sonar-large-128k-chat",
    "llama-3.1-sonar-huge-128k-online",
    "llama-3.1-70b-instruct",
]

GPT_AVAILABLE_MODELS = [
    "gpt-3.5-turbo",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-4o-mini",
    "gpt-4o",
    "o1-mini",
    "o1-preview",
]

ANTHROPIC_AVAILABLE_MODELS = [
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
    "claude-3-opus-latest",
    "claude-3-sonnet-latest",
    "claude-3-haiku-latest",
]

DEEPSEEK_AVAILABLE_MODELS = [
    "deepseek-coder",
]

QWEN_AVAILABLE_MODELS = [
    "qwen-max-0919",
    "qwen-max",
    "qwen-coder-turbo-latest",
    "qwen-long",
    # coder
    "qwen-coder-plus",
    "qwen2.5-coder-32b-instruct",
    # coder fast
    "qwen-coder-turbo",
    "qwen2.5-coder-7b-instruct",
]

OLLAMA_AVAILABLE_MODELS = [
    # general models
    "llama3.1",
    # code models
    "codegeex4",
    "codegeex4-9b-all-fp16",
    "codestral",
    "codestral:22b-v0.1-f16",
    "deepseek-coder-v2",
    "deepseek-coder-v2:16b-lite-instruct-fp16",
    "codegemma",
    "codeqwen",
]

import logging
logger = logging.getLogger("shared_logger")
logger.setLevel(logging.DEBUG)

import os
print(os.getcwd())
print(os.path.abspath(__file__))
cur_folder = os.path.dirname(os.path.abspath(__file__))
log_folder = os.path.join(cur_folder, "../logs")
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
# 创建一个文件处理器
file_handler = logging.FileHandler(os.path.join(log_folder, "translator.log"), encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

# 创建一个日志格式器并将其添加到处理器
formatter = logging.Formatter("%(asctime)s - %(threadName)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# 将处理器添加到 logger
logger.addHandler(file_handler)