from util.config import logger
from coder.analyzer import LangDesc
from coder.codebase import CodeBase
from util.config import (
    GPTModel,
    DeepseekModel,
    Language
)
class Translator(CodeBase):
    @property
    def system_prompt(self):
        return f"You are a translator from {self.src_lang} to {self.tar_lang}."
    
    def __init__(self, model=GPTModel.GPT35_TURBO, use_local=False, temperature=0.3, peft_model=""):
        super().__init__(model, use_local, temperature, peft_model)
    
    def translate(self, code_str: str) -> str:
        logger.info(f"Translating code from {self.src_lang} to {self.tar_lang}")
        prompt = f"""
You are about to translate the following {self.src_lang} code to {self.tar_lang} code.
Please follow the following rules:
1. Do not change the code structure.
2. Do not change the code semantics.
3. Do not change the code comments.
4. Do not change the code format.
5. Do not change the code variable names.
6. Do not change the code function names.
7. Do not change the code order.
8. Do not change the code logic.

Please translate the code step by step. Notice that replace the code comments with the corresponding {self.tar_lang.language} comments. replace the library that only exists in {self.src_lang.language} with the corresponding library that only exists in {self.tar_lang.language}, or directly simulate the library that only exists in {self.src_lang.language} in {self.tar_lang.language}.
Please output the translated code in the following format:
```{self.tar_lang.language}
int main() {{
    return 0;
}}
```

The following is the code to be translated:
{code_str}
"""
        resp = self.chat(prompt)
        # Split by ``` to get all code blocks
        code_blocks = resp.split("```")
        # Extract language and code from each block
        translated_blocks = []
        for i in range(1, len(code_blocks), 2):  # Skip non-code blocks
            if i < len(code_blocks):
                block = code_blocks[i]
                # Split first line as language, rest as code
                lines = block.split("\n", 1)
                if len(lines) > 1:
                    lang = lines[0].strip()
                    code = lines[1].strip()
                    translated_blocks.append([lang, code])
        # Print all translated code blocks
        translated_code = ""
        for i, (lang, code) in enumerate(translated_blocks):
            logger.info(f"Translation to {self.tar_lang} result {i+1}:\n{code}")
        return translated_blocks

class CUDA2CTranslator(Translator):
    
    def __init__(self, model=GPTModel.GPT35_TURBO, use_local=False, temperature=0.3, peft_model=""):
        # self.src_lang = "CUDA"
        # self.tar_lang = "C"
        self.src_lang = LangDesc(Language.CUDA, "12.1")
        self.tar_lang = LangDesc(Language.C, "11")
        super().__init__(model, use_local, temperature, peft_model)

class Python2CTranslator(Translator):
    def __init__(self, model=GPTModel.GPT35_TURBO, use_local=False, temperature=0.3, peft_model=""):
        # self.src_lang = "Python"
        # self.tar_lang = "C"
        self.src_lang = LangDesc(Language.PYTHON, "3.10")
        self.tar_lang = LangDesc(Language.C, "11")
        super().__init__(model, use_local, temperature, peft_model)

class CPP2PythonTranslator(Translator):
    def __init__(self, model=GPTModel.GPT35_TURBO, use_local=False, temperature=0.3, peft_model=""):
        self.src_lang = LangDesc(Language.CPP, "11")
        self.tar_lang = LangDesc(Language.PYTHON, "3.10")
        super().__init__(model, use_local, temperature, peft_model)

class LowerToAladdinTranslator(Translator):
    def __init__(self, model=DeepseekModel.CODER, use_local=False, temperature=0.3, peft_model=""):
        self.src_lang = LangDesc(Language.C, "11")
        self.tar_lang = LangDesc(Language.C, "11")
        super().__init__(model, use_local, temperature, peft_model)

    @property
    def system_prompt(self):
        return f"You are a coding assistant to add Aladdin Accelerator calls to the following {self.src_lang} code."

    def translate(self, code_str: str) -> str:
        logger.info(f"Lowering code from {self.src_lang} to Aladdin")
        from coder.prompts import direct_replace_to_aladdin
        prompt = direct_replace_to_aladdin.format(code_str=code_str)
        resp = self.chat(prompt)
        return resp
