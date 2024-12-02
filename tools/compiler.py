import os
import platform
import shutil
import datetime
import subprocess
from util.config import logger
import chardet
from clang.cindex import Index, CursorKind, TranslationUnit

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
            from util.config import DeepseekModel
            lower_to_aladdin_translator = LowerToAladdinTranslator(model=DeepseekModel.CODER)
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
                    elif current_section.startswith('match_gemv_fragments_'):
                        original_fragments.append(code)
                    elif current_section.startswith('replace_gemm_fragments_'):
                        replaced_fragments.append(code)
                    elif current_section.startswith('replace_gemv_fragments_'):
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

    def merge_files(self, src_dir: str) -> str:
        """合并所有C源文件和头文件到一个文件中"""
        # 存储所有的代码内容
        merged_content = []
        # 存储已处理的头文件，避免重复包含
        processed_headers = set()
        
        def process_includes(file_content: str) -> tuple[str, list[str]]:
            """处理include语句，返回处理后的内容和找到的本地头文件列表"""
            lines = file_content.split('\n')
            local_headers = []
            processed_lines = []
            
            for line in lines:
                if line.strip().startswith('#include'):
                    if '"' in line:  # 本地头文件
                        header = line.split('"')[1]
                        local_headers.append(header)
                    else:  # 系统头文件
                        processed_lines.append(line)
                else:
                    processed_lines.append(line)
                    
            return '\n'.join(processed_lines), local_headers

        def merge_file(file_path: str):
            """递归处理文件及其依赖"""
            if file_path in processed_headers:
                return
            
            try:
                # 明确指定使用 UTF-8 编码读取文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # 如果 UTF-8 失败，尝试其他编码
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    return
                
            # 处理include语句
            processed_content, headers = process_includes(content)
            
            # 递归处理本地头文件
            for header in headers:
                header_path = os.path.join(os.path.dirname(file_path), header)
                if os.path.exists(header_path) and header_path not in processed_headers:
                    merge_file(header_path)
                    
            processed_headers.add(file_path)
            merged_content.append(processed_content)

        # 查找所有.c文件
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.endswith('.c'):
                    file_path = os.path.join(root, file)
                    merge_file(file_path)
                    
        return '\n\n'.join(merged_content)

    def preprocess_code(self, src_dir: str, output_file: str = None):
        """预处理代码，合并文件并展开宏"""
        # 首先合并所有文件
        merged_code = self.merge_files(src_dir)
        
        if output_file is None:
            output_file = os.path.join(self.tmp_folder, "merged.c")
            
        # 写入合并后的代码
        with open(output_file, 'w') as f:
            f.write(merged_code)
            
        # 使用gcc预处理器展开宏
        preprocessed_file = output_file + ".preprocessed"
        os.system(f"gcc -E {output_file} -o {preprocessed_file}")
        
        return preprocessed_file

    def detect_encoding(self, file_path):
        with open(file_path, 'rb') as f:
            raw = f.read()
        return chardet.detect(raw)['encoding']

    def get_source_text(self, node, file_content=None):
        """从文件内容中获取节点的源代码文本"""
        try:
            if not node.extent or not node.extent.start or not node.extent.end:
                return ''
            
            if file_content is None:
                # 如果没有传入文件内容，尝试读取文件
                if node.location and node.location.file:
                    with open(str(node.location.file), 'r') as f:
                        file_content = f.read()
                else:
                    return ''
                
            start_pos = node.extent.start
            end_pos = node.extent.end
            
            if not start_pos.file or not end_pos.file:
                return ''
            
            # 获取行列信息
            start_line = start_pos.line - 1  # 转换为0基索引
            start_col = start_pos.column - 1
            end_line = end_pos.line - 1
            end_col = end_pos.column - 1
            
            # 分割成行
            lines = file_content.splitlines()
            
            # 如果是单行
            if start_line == end_line:
                if start_line < len(lines):
                    return lines[start_line][start_col:end_col]
                return ''
            
            # 如果是多行
            text = []
            for i in range(start_line, end_line + 1):
                if i >= len(lines):
                    break
                if i == start_line:
                    text.append(lines[i][start_col:])
                elif i == end_line:
                    text.append(lines[i][:end_col])
                else:
                    text.append(lines[i])
                
            return '\n'.join(text)
            
        except Exception as e:
            logger.debug(f"获取源代码文本时出错: {e}")
            return ''

    def parse_file(self, filename):
        abs_path = os.path.abspath(filename)
        if not os.path.exists(abs_path):
            logger.error(f"File not found: {abs_path}")
            return None
        
        # 读取文件内容
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
        except UnicodeDecodeError:
            # 如果UTF-8失败，尝试其他编码
            with open(abs_path, 'r', encoding='latin-1') as f:
                file_content = f.read()
        
        index = Index.create()
        try:
            # 添加更多编译选项以正确解析文件
            tu = index.parse(abs_path, 
                            args=['-x', 'c',  # 指定语言为C
                                  f'-I{os.path.dirname(abs_path)}',  # 添加源文件目录到include路径
                                  '-fparse-all-comments',  # 解析所有注释
                                  '-Wno-everything'],  # 禁用所有警告
                            options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
            
            if not tu:
                logger.error("Failed to parse translation unit")
                return None
            
            logger.info(f"Successfully parsed file: {abs_path}")
            
            def get_node_info(node):
                # 只处理来自目标文件的节点
                if (node.location.file and 
                    os.path.abspath(str(node.location.file)) == abs_path):
                    
                    # 判断变量是否为全局变量
                    is_global = (node.kind == CursorKind.VAR_DECL and 
                                node.semantic_parent and 
                                node.semantic_parent.kind == CursorKind.TRANSLATION_UNIT)
                    
                    info = {
                        'spelling': node.spelling,
                        'type': str(node.type.spelling) if hasattr(node, 'type') else '',
                        'file': str(node.location.file) if node.location and node.location.file else None,
                        'line': node.location.line if node.location else None,
                        'column': node.location.column if node.location else None,
                        'extent': None,
                        'usr': node.get_usr() if hasattr(node, 'get_usr') else '',
                        'source': self.get_source_text(node, file_content),
                        'is_global': is_global  # 添加是否为全局变量的标记
                    }
                    
                    if node.extent:
                        info['extent'] = {
                            'start': (node.extent.start.line, node.extent.start.column) if node.extent.start else None,
                            'end': (node.extent.end.line, node.extent.end.column) if node.extent.end else None
                        }
                        
                    return info
                return None

            # 用于存储解析结果的数据结构
            parsed_info = {
                'includes': [],
                'structs': [],
                'functions': [],
                'variables': [],
                'macros': [],
                'source_map': {}
            }
            
            def visit_nodes(node, depth=0):
                try:
                    # 只处理来自目标文件的节点
                    if (node.location.file and 
                        os.path.abspath(str(node.location.file)) == abs_path):
                        
                        info = get_node_info(node)
                        if info:
                            logger.debug('\n'.join([
                                'Node Info:',
                                f'  Spelling: {info["spelling"]}',
                                f'  Type: {info["type"]}',
                                f'  File: {info["file"]}',
                                f'  Location: Line {info["line"]}, Column {info["column"]}',
                                f'  Extent: {info["extent"]}',
                                f'  USR: {info["usr"]}',
                                f'  Source: {info["source"]}'
                            ]))
                            
                            # 根据节点类型进行分类存储
                            if node.kind == CursorKind.INCLUSION_DIRECTIVE:
                                parsed_info['includes'].append(info)
                            elif node.kind == CursorKind.FUNCTION_DECL:
                                parsed_info['functions'].append(info)
                            elif node.kind == CursorKind.VAR_DECL:
                                parsed_info['variables'].append(info)
                            elif node.kind == CursorKind.STRUCT_DECL:
                                parsed_info['structs'].append(info)
                            elif node.kind == CursorKind.MACRO_DEFINITION:
                                parsed_info['macros'].append(info)
                
                    # 递归访问子节点
                    for child in node.get_children():
                        visit_nodes(child, depth + 1)
                        
                except Exception as e:
                    logger.error(f"解析节点时出错: {str(e)}")
                    logger.info("\n")
            
            visit_nodes(tu.cursor)
            return parsed_info
            
        except Exception as e:
            logger.error(f"解析文件时出错: {e}")
            return None

    def reconstruct_source(self, parsed_info):
        """根据解析信息重建源代码"""
        if not parsed_info:
            logger.error("No parsed information available")
            return ""
        
        source_lines = []
        
        # 添加包含指令
        for include in parsed_info['includes']:
            # 使用 source 而不是 name，因为现在我们存储的是完整的 include 信息
            if 'source' in include:
                source_lines.append(include['source'])
        
        source_lines.append('')  # 空行分隔
        
        # 添加宏定义
        for macro in parsed_info['macros']:
            if 'source' in macro:
                source_lines.append(macro['source'])
        
        source_lines.append('')  # 空行分隔
        
        # 添加结构体定义
        for struct in parsed_info['structs']:
            if 'source' in struct:
                source_lines.append(struct['source'])
                source_lines.append('')
        
        # 添加全局变量
        for var in parsed_info['variables']:
            if 'source' in var:
                source_lines.append(var['source'])
        
        source_lines.append('')  # 空行分隔
        
        # 添加函数
        for func in parsed_info['functions']:
            if 'source' in func:
                source_lines.append(func['source'])
                source_lines.append('')
        
        return '\n'.join(source_lines)
