from util.config import Language
import os, platform, subprocess
from coder.analyzer import Context

class Executor:
    def __init__(self, target_language: Language):
        self.target_language = target_language

    @property
    def target_language(self) -> Language:
        return self.target_language

    """
    执行可执行文件并返回输出结果
    
    Args:
        exec_folder (str): 可执行文件所在文件夹路径
        exec_target_name (str): 可执行文件名称（不含扩展名）
        version_id (str): 版本标识
        context (Context, optional): 执行上下文
        
    Returns:
        dict: 包含执行结果的字典，格式为:
            {
                'stdout': str,  # 标准输出
                'stderr': str,  # 标准错误
                'returncode': int  # 返回码
            }
    """
    def execute(self, exec_folder: str, exec_target_name: str, version_id: str, context: Context = None) -> dict:
        
        # 验证可执行文件
        assert os.path.exists(exec_file_path), f"可执行文件 {exec_file_path} 不存在"
        assert os.path.isfile(exec_file_path), f"路径 {exec_file_path} 不是一个文件"
        assert os.access(exec_file_path, os.X_OK), f"文件 {exec_file_path} 没有执行权限"
        assert exec_target_name, "需要提供可执行文件名称"
        system = platform.system()
        
        # 根据不同操作系统和语言确定可执行文件路径和扩展名
        if system == "Windows":
            path_separator = "\\"
            extensions = {
                Language.CPP: ".exe",
                Language.C: ".exe",
                Language.PYTHON: ".py",
                Language.JAVA: ".class",
                Language.CUDA: ".exe"
            }
        else:  # Linux 或 MacOS
            path_separator = "/"
            extensions = {
                Language.CPP: "",
                Language.C: "",
                Language.PYTHON: ".py",
                Language.JAVA: ".class",
                Language.CUDA: ""
            }
        
        # 构建完整的可执行文件路径
        extension = extensions.get(self.target_language, "")
        exec_file_path = f"{exec_folder}{path_separator}{exec_target_name}{extension}"
        
        # 根据不同语言构建执行命令
        command = exec_file_path
        if self.target_language == Language.PYTHON:
            command = f"python {exec_file_path}"
        elif self.target_language == Language.JAVA:
            command = f"java -cp {exec_folder} {exec_target_name}"

        try:
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                capture_output=True,
                timeout=30  # 30秒超时限制
            )
            
            return {
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                'stdout': '',
                'stderr': '执行超时',
                'returncode': -1
            }
        except Exception as e:
            return {
                'stdout': '',
                'stderr': str(e),
                'returncode': -1
            }
