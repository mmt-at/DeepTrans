from util.config import logger
import os
import torch
import anthropic
from openai import OpenAI
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from util.config import (
    # api keys
    PPLX_API_KEY,
    QWEN_API_KEY,
    ANTHROPIC_API_KEY,
    DEEPSEEK_API_KEY,
    OPENAI_API_KEY,
    # available models
    PPLX_AVAILABLE_MODELS,
    GPT_AVAILABLE_MODELS,
    ANTHROPIC_AVAILABLE_MODELS,
    DEEPSEEK_AVAILABLE_MODELS,
    QWEN_AVAILABLE_MODELS,
    OLLAMA_AVAILABLE_MODELS,
)
import tiktoken
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
)

def num_token_from_string(
    string: str, encoding_name: str = "gpt-3.5-turbo-0613"
) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


class ChatBase:
    @property
    def system_prompt(self):
        return """You are a helpful assistant. You will follow the user's instructions to complete the task."""

    @property
    def initial_messages(self):
        if self.model == "o1-preview":
            return []
        return [
            {
                "role": "system",
                "content": self.system_prompt,
            },
        ]

    def __init__(
        self,
        model="mixtral-8x7b-instruct",
        use_local=False,
        temperature=0.3,
        peft_model="",
    ):
        if (
            model not in PPLX_AVAILABLE_MODELS
            and model not in GPT_AVAILABLE_MODELS
            and model not in ANTHROPIC_AVAILABLE_MODELS
            and model not in DEEPSEEK_AVAILABLE_MODELS
            and model not in QWEN_AVAILABLE_MODELS
            and model not in OLLAMA_AVAILABLE_MODELS
            and use_local is False
        ):
            logger.error(f"Model {model} is not available!")
            exit(1)
        self.use_local = use_local
        self.model = model
        self.temperature = temperature
        if self.model in GPT_AVAILABLE_MODELS:
            self.client = self.gpt_client = OpenAI(api_key=OPENAI_API_KEY)
        elif self.model in ANTHROPIC_AVAILABLE_MODELS:
            self.client = self.claude_client = anthropic.Anthropic(
                api_key=ANTHROPIC_API_KEY
            )
        elif self.model in DEEPSEEK_AVAILABLE_MODELS:
            self.client = self.deepseek_client = OpenAI(
                api_key=DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com/beta",  # max 8192 output
            )
        elif self.model in PPLX_AVAILABLE_MODELS:
            self.client = self.pplx_client = OpenAI(
                api_key=PPLX_API_KEY, base_url="https://api.perplexity.ai"
            )
        elif self.model in QWEN_AVAILABLE_MODELS:
            self.client = self.qwen_client = OpenAI(
                api_key=QWEN_API_KEY,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        elif self.use_local:
            self.use_local = True
            self.tokenizer = AutoTokenizer.from_pretrained(
                model, trust_remote_code=True
            )
            self.local_model = AutoModelForCausalLM.from_pretrained(
                model,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            if peft_model != "":
                self.local_model = PeftModel.from_pretrained(
                    self.local_model, peft_model
                )
                self.is_finetuned = True
            else:
                self.is_finetuned = False
            self.client = self.local_model
        else:
            self.client = self.ollama_client = OpenAI(
                base_url="http://localhost:11434/v1/",
                api_key="ollama",  # required, but unused
            )
        self.message_reset()

    """
    reset the messages, only keep the system prompt, which clear the chat history
    """

    def message_reset(self):
        self.messages = self.initial_messages
        if self.model in ANTHROPIC_AVAILABLE_MODELS:
            self.messages = []

    def chat(self, user_input, temperature=0.3):
        if user_input is None:
            user_input = input("User: \n")
            logger.info(f"stdin user input: \n{user_input}")
        self.messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )
        query_size = num_tokens_from_messages(self.messages)
        logger.info(f"current LLM prompt size: {query_size}")
        logger.debug(f"###current LLM prompt: {self.messages}")
        if query_size >= 8192:
            logger.warning(
                "LLM prompt size exceeds the limit 8192, will truncate the prompt."
            )
        if self.model in ANTHROPIC_AVAILABLE_MODELS:
            anthropic_client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            message = anthropic_client.messages.create(
                model=self.model,
                system=self.system_prompt,
                max_tokens=8192,
                temperature=temperature,
                messages=self.messages,
            )
            rsp_content = message.content[0].text
        else:
            if self.model in ["o1-preview", "o1-mini"]:
                temperature = 1
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=self.messages,
            )
            rsp_content = response.choices[0].message.content
        logger.debug(f"###LLM response: \n{rsp_content}")
        self.messages.append(
            {
                "role": "assistant",
                "content": rsp_content,
            }
        )
        return rsp_content

    def response(self):
        return self.messages[-1]["content"]

    def paraphrase(self, text):
        task_description = "paraphrase the following text, make sure it is well written and syntax correct."
        prompt = f"{task_description}\n{text}"
        self.chat(user_input=prompt)
        paraphrased_text = self.messages[-1]["content"]
        logger.info(f"Paraphrased text: \n{paraphrased_text}")
        return paraphrased_text
