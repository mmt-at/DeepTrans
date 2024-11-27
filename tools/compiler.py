import os
import platform
import shutil
import datetime
import subprocess
from util.config import logger

class CompilerCaller:
    def __init__(self, language: str):
        self.language = language
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.tmp_folder = os.path.join(current_dir, "..", "data", "tmp")
        if not os.path.exists(self.tmp_folder):
            os.makedirs(self.tmp_folder)

    def extract_and_expand_function(self, header_file, source_file, function_name):
        # 读取头文件并提取函数定义
        with open(header_file, 'r') as file:
            lines = file.readlines()
        
        function_definition = ""
        for line in lines:
            if function_name in line:
                function_definition = line.strip()
                break
        
        # 将函数定义展开到源文件中
        with open(source_file, 'r') as file:
            source_lines = file.readlines()
        
        expanded_source = []
        for line in source_lines:
            if function_name in line:
                expanded_source.append(function_definition + "\n")
            expanded_source.append(line)
        
        return expanded_source

    def run_executable(self, build_dir: str = None, function_name: str = None):
        if build_dir is None:
            build_dir = self.build_dir
        executable_path = self.executable_path
        
        if not os.path.exists(executable_path):
            logger.error(f"Executable not found at {executable_path}")
            return False
            
        try:
            result = subprocess.run([executable_path], capture_output=True, text=True)
            logger.info("Execution output:")
            logger.info(result.stdout)
            if result.stderr:
                logger.error("Execution errors:")
                logger.error(result.stderr)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error running executable: {e}")
            return False

    def build(self, position: str = "data", level: str = "repo", lang: str = "c", name: str = "test", target_func_name: str = "test", inplace: bool = False):
        src_from_dir = os.path.join(position, level, lang, name)
        if level == "repo":
            if inplace:
                src_to_dir = src_from_dir
            else:
                # 获取当前时间
                date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                src_to_dir = os.path.abspath(os.path.join(self.tmp_folder, level, lang, name, f"{date_time}"))
                os.makedirs(src_to_dir, exist_ok=True)
            
            self.src_to_dir = src_to_dir
            if target_func_name is None:
                logger.error("No target function name provided")
                exit(0)
            
            # 列出src_dir下所有.c和.h文件
            src_files = {}
            for root, dirs, files in os.walk(src_from_dir):
                for file in files:
                    if file.endswith(('.c', '.h')):
                        # 使用绝对路径计算相对路径
                        abs_file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(abs_file_path, src_from_dir)
                        src_files[rel_path] = file

            logger.info(f"Found following .c and .h files in {src_from_dir}:")
            for relpath, name_with_suffix in src_files.items():
                logger.info(f"{relpath}\t{name_with_suffix}")
            
            # 复制文件
            for relpath, _ in src_files.items():
                src_path = os.path.join(src_from_dir, relpath)
                dst_path = os.path.join(src_to_dir, relpath)
                # 确保目标目录存在
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)

            # 编译代码
            build_dir = os.path.join(src_to_dir, "build")
            os.makedirs(build_dir, exist_ok=True)
            self.build_dir = build_dir
            
            # 获取所有包含.h文件的目录
            include_dirs = set()
            for relpath, _ in src_files.items():
                if relpath.endswith('.h'):
                    include_dirs.add(os.path.dirname(os.path.join(src_to_dir, relpath)))

            # 构建gcc命令
            gcc_cmd = ["gcc",
                *[os.path.join(src_to_dir, relpath) for relpath, _ in src_files.items() if relpath.endswith('.c')],
                *[item for include_dir in include_dirs for item in ("-I", include_dir)],  # 添加所有包含.h文件的目录
                "-o", os.path.join(build_dir, f"{target_func_name}")
            ]
            self.executable_path = os.path.join(build_dir, f"{target_func_name}")
            if platform.system() == "Windows":
                self.executable_path += ".exe"
            logger.info(f"Executing gcc command: {' '.join(gcc_cmd)}")

            compile_result = subprocess.run(
                gcc_cmd,
                capture_output=True,
                text=True
            )
            
            # 打印编译错误信息（如果有）
            if compile_result.returncode != 0:
                logger.error("Compilation failed:")
                logger.error(compile_result.stderr)
                
            return src_to_dir
            
        elif name == "gaussian-splatting-c":
            repo_path = os.path.join(position, level, lang, name)
            os.system(f"pip install -e {repo_path}")

    def lowering(self, position: str = "data", level: str = "repo", lang: str = "c", name: str = "test", target_func_name: str = "test", inplace: bool = False):
        from coder.prompts import direct_replace_to_aladdin
        src_from_dir = os.path.join(position, level, lang, name)
        src_to_dir = os.path.join(self.tmp_folder, level, lang, name)
        os.makedirs(src_to_dir, exist_ok=True)
        src_files = {}
        for root, dirs, files in os.walk(src_from_dir):
            for file in files:
                if file.endswith(('.c', '.h')):
                    src_files[os.path.relpath(os.path.join(root, file), src_from_dir)] = file
        
        # 复制文件
        for relpath, _ in src_files.items():
            src_path = os.path.join(src_from_dir, relpath)
            dst_path = os.path.join(src_to_dir, relpath)
            # 确保目标目录存在
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)

        original_fragments_dict = {}
        complete_code_dict = {}
        replaced_fragments_dict = {}
        replaced_complete_code_dict = {}
        for relpath, _ in src_files.items():
            logger.debug(f"Processing file: {relpath}")
            with open(os.path.join(src_to_dir, relpath), 'r', encoding='utf-8') as file:
                code_str = file.read()
            from coder.translator import LowerToAladdinTranslator
            lower_to_aladdin_translator = LowerToAladdinTranslator()
            resp = lower_to_aladdin_translator.translate(code_str)
            
            # 解析翻译结果
            original_fragments = []
            replaced_fragments = []
            current_section = None
            code_content = ""
            for line in resp.split('\n'):
                # 检测section标记
                import re
                if re.match(r'^\s*#\s+', line):
                    current_section = re.sub(r'^\s*#\s+', '', line).strip()
                    continue

                # 跳过空行
                if not line.strip():
                    continue

                # 如果是代码块开始
                if line.startswith(f'```{lang}'):
                    code_content = []
                    continue

                # 如果是代码块结束
                if line.endswith('```'):
                    code = '\n'.join(code_content)
                    if current_section == 'include_fragments':
                        original_fragments.append(code)
                        replaced_fragments.append(code)
                    elif current_section == 'function_fragments':
                        original_fragments.append(code)
                        replaced_fragments.append(code)
                    elif current_section.startswith('match_gemm_fragments_'):
                        original_fragments.append(code)
                    elif current_section.startswith('replace_gemm_fragments_'):
                        replaced_fragments.append(code)
                    elif current_section == 'no_match_function':
                        original_fragments.append(code)
                        replaced_fragments.append(code)
                    continue
                    
                # 收集代码内容
                if current_section:
                    code_content.append(line)

            # 生成完整代码
            complete_code = '\n'.join(original_fragments)
            replaced_complete_code = '\n'.join(replaced_fragments)
            original_fragments_dict[relpath] = original_fragments
            complete_code_dict[relpath] = complete_code
            replaced_fragments_dict[relpath] = replaced_fragments
            replaced_complete_code_dict[relpath] = replaced_complete_code
            logger.debug(f"original_fragments: {original_fragments}")
            logger.debug(f"complete_code: {complete_code}")
            logger.debug(f"original_fragments_dict: {original_fragments_dict}")
            logger.debug(f"replaced_fragments: {replaced_fragments}")
            logger.debug(f"replaced_complete_code: {replaced_complete_code}")
            logger.debug(f"replaced_fragments_dict: {replaced_fragments_dict}")
            logger.debug(f"replaced_complete_code_dict: {replaced_complete_code_dict}")
            with open(os.path.join(src_to_dir, relpath), 'w', encoding='utf-8') as file:
                file.write(resp)
        return src_to_dir, original_fragments_dict, complete_code_dict, replaced_fragments_dict, replaced_complete_code_dict

    def compile_code(self, code: str):
        pass
