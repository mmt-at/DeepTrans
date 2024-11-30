from coder.codebase import CodeBase
from typing import List
from coder.backend import TemplateFiller
from util.config import Language

class SlicingAgent(CodeBase):
    def __init__(
        self,
        model="mixtral-8x7b-instruct",
        use_local=False,
        temperature=0.3,
        peft_model="",
    ):
        super().__init__(model, use_local, temperature, peft_model)

class CompilerAgent(CodeBase):
    def compile(self, context: "Context") -> str:
        pass

class Location:
    def __init__(self, file_path: str, start_line: int, end_line: int):
        self.file_path = file_path
        self.start_line = start_line
        if not end_line:
            self.end_line = start_line
        else:
            self.end_line = end_line

class LangDesc:
    def __init__(self, language: Language, version_id: str):
        self.language = language
        self.version_id = version_id

    def __str__(self):
        return f"{self.language}{self.version_id}"
    
    def __repr__(self):
        return f"{self.language}{self.version_id}"

class Node:
    pass

class CodeFragment(Node):
    def __init__(self, lang_desc: LangDesc, location: Location, code_str: str):
        if lang_desc.language not in [Language.CPP, Language.PYTHON, Language.JAVA, Language.CUDA, Language.C]:
            raise ValueError(f"Language {lang_desc.language} is not supported")
        self.lang_desc = lang_desc
        self.location = location
        self.code_str = code_str

class CodeSnippet(Node):
    # Addressing Aladdin's insertion, the code to be inserted is referred to as a snippet    
    def __init__(self, tar_rep_code_frags: List[CodeFragment], _method: str = "Aladdin"):
        self.tar_rep_code_frags = tar_rep_code_frags
        self._method = _method

    @property
    def tar_rep_code_frags(self) -> List[CodeFragment]:
        return self.tar_rep_code_frags

    def accelerator_template_simple_matmul4x4(self):
        config_dict = {
            "data_type": "float16",
            "input_dims": [16, 16],
            "weight_dims": [16, 16],
            "check_code": "",
        }
        code_template = TemplateFiller.aladdin_simple_matmul_template(**config_dict)
        return code_template

class SymbalTable:
    pass

class Context:
    def __init__(self, folder_path: str, file_path: str, code_str: str, ):
        self.folder_path = folder_path
        self.file_path = file_path
        self.code_str = code_str
        self.symbal_table = SymbalTable()

class Tracer:
    pass

class Executor:
    def __init__(self, target_language: Language):
        self.target_language = target_language

    @property
    def target_language(self) -> Language:
        return self.target_language

    def execute(self, context: Context) -> str:
        pass

class PythonInterpreter(Executor):
    pass

class CRunner(Executor):
    pass

class CPPRunner(Executor):
    pass

class JavaRunner(Executor):
    pass

class CUDARunner(Executor):
    pass

class Validator:
    def __init__(self, model: str):
        self.model = model

    def validate(self, text: str) -> str:
        pass