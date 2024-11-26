import os

class CompilerCaller:
    def __init__(self, language: str):
        self.language = language
        current_dir = os.path.dirname(os.path.abspath(__file__))
        code_folder = os.path.join(current_dir, "../tmp/code")
        if not os.path.exists(code_folder):
            os.makedirs(code_folder)

    def compile_code(self, code: str):
        pass