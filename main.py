import gradio as gr
import asyncio
from pathlib import Path
import json
import logging
from modules.api_config import APIConfig
from modules.text_processor import TextProcessor
from modules.image_processor import ImageProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 默认API设置
DEFAULT_API_KEY = ""
DEFAULT_BASE_URL = ""
DEFAULT_MODEL_NAME = "gpt-4o-mini"

class DataMakerApp:
    def __init__(self):
        self.api_config = APIConfig()
        # 设置默认API配置
        self.api_config.set_config(DEFAULT_API_KEY, DEFAULT_BASE_URL, DEFAULT_MODEL_NAME)
        self.text_processor = TextProcessor(self.api_config)
        self.image_processor = ImageProcessor(self.api_config)
        self.processed_text_data = []
        self.processed_image_data = []
        
        # 创建输出目录结构
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        self.input_figure_dir = self.output_dir / "input_figure_dir"
        self.input_figure_dir.mkdir(exist_ok=True)
        logger.info("应用初始化完成")
        
    def test_api(self, api_key: str, base_url: str, model_name: str) -> str:
        """测试API连接"""
        try:
            self.api_config.set_config(api_key, base_url, model_name)
            success, message = self.api_config.test_connection()
            logger.info(f"API测试结果: {message}")
            return message
        except Exception as e:
            error_msg = f"API测试失败: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def process_text_file(self, file, text_analysis_prompt: str, title_prompt: str, format_prompt: str) -> str:
        """处理文本文件"""
        if not file:
            return "请先上传文本文件"
            
        try:
            logger.info(f"开始处理文本文件: {file.name if hasattr(file, 'name') else 'unknown'}")
            # 更新提示词
            self.text_processor.update_prompts(text_analysis_prompt, title_prompt, format_prompt)
            
            # 读取文件内容
            content = None
            if hasattr(file, 'name'):  # 如果是文件对象
                with open(file.name, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                return "文件格式错误"
            
            if not content:
                return "文件内容为空"

            # 创建新的事件循环来处理异步操作
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                result = new_loop.run_until_complete(self.text_processor.process_text(content))
            finally:
                new_loop.close()
                asyncio.set_event_loop(None)
            
            if result["success"]:
                self.processed_text_data.append(result["data"])
                success_msg = f"处理成功！当前已处理 {len(self.processed_text_data)} 条数据\n数据内容：{json.dumps(result['data'], ensure_ascii=False, indent=2)}"
                logger.info("文本处理成功")
                return success_msg
            else:
                error_msg = f"处理失败：{result['error']}"
                logger.error(error_msg)
                return error_msg

        except UnicodeDecodeError:
            error_msg = "文件编码错误，请确保文件为UTF-8编码的文本文件"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"处理失败：{str(e)}"
            logger.error(f"文本处理异常: {str(e)}", exc_info=True)
            return error_msg

    def save_text_dataset(self, output_path: str) -> str:
        """保存文本数据集"""
        try:
            if not self.processed_text_data:
                return "没有可保存的数据"
            
            # 确保使用output目录
            full_path = self.output_dir / output_path
            success, message = self.text_processor.save_dataset(
                self.processed_text_data,
                str(full_path)
            )
            if success:
                self.processed_text_data = []  # 清空已处理数据
                logger.info(f"文本数据集保存成功: {full_path}")
            else:
                logger.error(f"文本数据集保存失败: {message}")
            return message
        except Exception as e:
            error_msg = f"保存失败：{str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    def process_images(self, files, prompt: str, progress=gr.Progress()):
        """处理图片"""
        if not files:
            return [], "请先上传图片文件", gr.update(visible=True), "等待处理"
            
        try:
            logger.info(f"开始处理图片文件，数量: {len(files)}")
            progress(0, desc="准备处理图片...")
            
            # 获取文件路径并复制图片到input_figure_dir
            image_paths = []
            for file in files:
                if hasattr(file, 'name'):
                    # 复制图片到input_figure_dir并获取新路径
                    new_path = self.copy_image_to_input_dir(file.name)
                    image_paths.append(new_path)
                else:
                    logger.warning(f"跳过无效文件: {file}")
            
            if not image_paths:
                return [[]], "没有有效的图片文件", gr.update(visible=True), "处理失败"
            
            # 创建新的事件循环来处理异步操作
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                
                # 更新进度显示
                total_images = len(image_paths)
                async def process_with_progress():
                    results = []
                    for idx, image_path in enumerate(image_paths):
                        progress((idx / total_images) * 0.9, f"正在处理第 {idx + 1}/{total_images} 张图片...")
                        result = await self.image_processor.process_image(image_path, prompt)
                        if result["success"]:
                            # 使用绝对路径
                            abs_path = str(Path(image_path).absolute())
                            results.append({
                                "id": idx + 1,
                                "image_path": abs_path,
                                "description": result["description"]
                            })
                    return results
                
                results = new_loop.run_until_complete(process_with_progress())
                logger.info(f"成功处理 {len(results)} 张图片")
                progress(0.95, desc="正在格式化结果...")
            except Exception as e:
                logger.error(f"批量处理失败: {str(e)}", exc_info=True)
                return [[]], f"批量处理失败: {str(e)}", gr.update(visible=True), "处理失败"
            finally:
                new_loop.close()
                asyncio.set_event_loop(None)
                
            if not results:
                return [[]], "没有成功处理的图片", gr.update(visible=True), "处理失败"
                
            # 转换为DataFrame格式
            data = []
            for item in results:
                try:
                    data.append([
                        item["id"],
                        item["image_path"],
                        item["description"]
                    ])
                    # 同时保存到processed_image_data中
                    self.processed_image_data.append(item)
                except KeyError as e:
                    logger.error(f"数据格式错误: {str(e)}")
                    continue
            
            if not data:
                return [[]], "处理结果格式化失败", gr.update(visible=True), "处理失败"
                
            progress(1.0, desc="处理完成")
            logger.info(f"成功生成 {len(data)} 条描述数据")
            return data, "", gr.update(visible=False), f"成功处理 {len(data)} 张图片"
            
        except Exception as e:
            error_msg = f"处理失败：{str(e)}"
            logger.error(f"图片处理异常: {str(e)}", exc_info=True)
            return [[]], error_msg, gr.update(visible=True), "处理失败"
            
    def save_image_dataset(self, output_path: str) -> tuple:
        """保存图片数据集"""
        try:
            if not self.processed_image_data:
                return "没有可保存的数据", "等待处理"
            
            # 确保使用output目录
            full_path = self.output_dir / output_path
            
            # 检查是否存在现有文件
            existing_data = []
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:  # 只有当文件不为空时才尝试解析
                            existing_data = json.loads(content)
                            # 确保所有ID都是整数
                            for item in existing_data:
                                try:
                                    item['id'] = int(str(item['id']).strip())
                                except (ValueError, TypeError):
                                    # 如果转换失败，给一个默认值
                                    item['id'] = 0
                            logger.info(f"读取到现有数据 {len(existing_data)} 条")
                except json.JSONDecodeError:
                    logger.warning("文件为空或格式错误，将创建新文件")
                except Exception as e:
                    logger.error(f"读取现有文件失败: {str(e)}")
                    return f"读取现有文件失败: {str(e)}", "保存失败"
            
            # 过滤掉空数据
            self.processed_image_data = [item for item in self.processed_image_data 
                                       if all(str(v).strip() != "" for v in item.values())]
            
            # 合并数据
            if existing_data:
                try:
                    # 获取现有的最大ID
                    max_id = max(int(str(item['id']).strip()) for item in existing_data if str(item['id']).strip().isdigit())
                except (ValueError, TypeError):
                    max_id = 0
                
                # 更新新数据的ID
                for item in self.processed_image_data:
                    max_id += 1
                    item['id'] = max_id
                # 合并数据
                all_data = existing_data + self.processed_image_data
                logger.info(f"合并后共 {len(all_data)} 条数据")
            else:
                # 如果没有现有数据，直接使用处理好的数据
                all_data = self.processed_image_data
            
            # 保存数据
            try:
                with open(full_path, 'w', encoding='utf-8') as f:
                    json.dump(all_data, f, ensure_ascii=False, indent=2)
                self.processed_image_data = []  # 清空已处理数据
                logger.info(f"图片数据集保存成功: {full_path}")
                return "", f"数据集已保存到: {output_path}"
            except Exception as e:
                logger.error(f"保存文件失败: {str(e)}")
                return f"保存文件失败: {str(e)}", "保存失败"
                
        except Exception as e:
            error_msg = f"保存失败：{str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg, "保存失败"
        
    def create_ui(self):
        """创建Gradio界面"""
        with gr.Blocks(title="数据集创建工具") as app:
            gr.Markdown("# 数据集创建工具")
            
            with gr.Tabs():
                # API设置标签页
                with gr.Tab("API设置"):
                    api_key = gr.Textbox(
                        label="OpenAI API Key",
                        type="password",
                        value=DEFAULT_API_KEY
                    )
                    base_url = gr.Textbox(
                        label="Base URL（可选）",
                        value=DEFAULT_BASE_URL
                    )
                    model_name = gr.Textbox(
                        label="模型名称",
                        value=DEFAULT_MODEL_NAME
                    )
                    test_btn = gr.Button("测试API连接")
                    test_output = gr.Textbox(label="测试结果")
                    
                    test_btn.click(
                        fn=self.test_api,
                        inputs=[api_key, base_url, model_name],
                        outputs=test_output
                    )
                
                # 文本处理标签页
                with gr.Tab("文本处理"):
                    with gr.Row():
                        with gr.Column():
                            text_analysis_prompt = gr.Textbox(
                                label="文本分析Agent提示词",
                                value=self.text_processor.text_analysis_prompt,
                                lines=5
                            )
                            title_prompt = gr.Textbox(
                                label="标题生成Agent提示词",
                                value=self.text_processor.title_generation_prompt,
                                lines=5
                            )
                            format_prompt = gr.Textbox(
                                label="格式化Agent提示词",
                                value=self.text_processor.format_prompt,
                                lines=5
                            )
                        
                        with gr.Column():
                            text_output = gr.Textbox(label="处理结果", lines=10)
                    
                    with gr.Row():
                        text_file = gr.File(label="上传文本文件")
                        process_text_btn = gr.Button("处理文本")
                        save_text_btn = gr.Button("保存数据集")
                        text_save_path = gr.Textbox(
                            label="保存路径",
                            value="text_dataset.json"
                        )
                    
                    process_text_btn.click(
                        fn=self.process_text_file,
                        inputs=[text_file, text_analysis_prompt, title_prompt, format_prompt],
                        outputs=text_output
                    )
                    
                    save_text_btn.click(
                        fn=self.save_text_dataset,
                        inputs=[text_save_path],
                        outputs=text_output
                    )
                
                # 图像处理标签页
                with gr.Tab("图像处理"):
                    with gr.Row():
                        with gr.Column():
                            image_files = gr.Files(
                                label="上传图片文件",
                                file_types=["image"],
                                type="filepath"
                            )
                            image_preview = gr.Gallery(
                                label="图片预览",
                                show_label=True,
                                elem_id="gallery",
                                columns=4,
                                height=300
                            )
                            # 添加图片预览更新
                            image_files.change(
                                fn=lambda files: [file.name for file in files] if files else None,
                                inputs=image_files,
                                outputs=image_preview
                            )
                        
                        with gr.Column():
                            # 添加状态显示
                            status_output = gr.Textbox(
                                label="处理状态",
                                value="等待处理",
                                interactive=False
                            )
                            with gr.Group():  # 使用Box包装Dataframe，避免显示进度条
                                image_output = gr.Dataframe(
                                    headers=["编号", "图片路径", "描述"],
                                    label="处理结果",
                                    wrap=True,
                                    interactive=True
                                )
                    
                    image_prompt = gr.Textbox(
                        label="图片描述提示词",
                        value=self.image_processor.default_prompt,
                        lines=3
                    )
                    
                    with gr.Row():
                        process_images_btn = gr.Button("批量生成描述", variant="primary")
                        save_images_btn = gr.Button("保存数据集", variant="secondary")
                        image_save_path = gr.Textbox(
                            label="保存路径",
                            value="image_dataset.json"
                        )
                    
                    # 添加错误信息显示
                    error_output = gr.Textbox(
                        label="错误信息",
                        visible=False,
                        show_label=True
                    )
                    
                    # 添加数据更新事件
                    image_output.change(
                        fn=self.update_image_data,
                        inputs=[image_output],
                        outputs=[error_output, status_output]
                    )
                    
                    # 添加CSS样式来隐藏DataFrame的进度条
                    gr.HTML("""
                        <style>
                            .gradio-container .gradio-dataframe .progress {
                                display: none !important;
                            }
                        </style>
                    """)
                    
                    process_images_btn.click(
                        fn=self.process_images,
                        inputs=[image_files, image_prompt],
                        outputs=[image_output, error_output, error_output, status_output],
                        show_progress=True
                    )
                    
                    # 修改保存按钮的回调函数输出
                    save_images_btn.click(
                        fn=self.save_image_dataset,
                        inputs=[image_save_path],
                        outputs=[error_output, status_output]
                    )
            
            return app

    def update_image_data(self, data):
        """更新图片处理数据"""
        try:
            # 检查数据是否为空
            if hasattr(data, 'empty') and data.empty:
                logger.warning("没有要更新的数据")
                return "", "等待处理"
            
            # 将DataFrame数据转换为列表
            if hasattr(data, 'values'):  # 如果是DataFrame
                new_rows = data.values.tolist()
            else:  # 如果是列表
                new_rows = data
            
            # 过滤掉空行
            new_rows = [row for row in new_rows if isinstance(row, (list, tuple)) and 
                       len(row) >= 3 and all(x is not None and str(x).strip() != "" for x in row)]
            
            # 比较新旧数据，只更新修改过的数据
            updated_data = []
            for row in new_rows:
                try:
                    # 确保ID是整数
                    row_id = int(row[0]) if isinstance(row[0], (int, str)) else 0
                    
                    # 检查是否是已存在的数据
                    existing_item = next(
                        (item for item in self.processed_image_data if int(item["id"]) == row_id), 
                        None
                    )
                    
                    if existing_item:
                        # 如果数据有变化，则更新
                        if (existing_item["image_path"] != str(row[1]) or 
                            existing_item["description"] != str(row[2])):
                            existing_item.update({
                                "image_path": str(row[1]),
                                "description": str(row[2])
                            })
                            updated_data.append(existing_item)
                    else:
                        # 新数据，添加到列表
                        new_item = {
                            "id": row_id,
                            "image_path": str(row[1]),
                            "description": str(row[2])
                        }
                        self.processed_image_data.append(new_item)
                        updated_data.append(new_item)
                except Exception as e:
                    logger.error(f"处理数据行时出错: {str(e)}")
                    continue
            
            if not updated_data:
                return "", "没有数据需要更新"
                
            logger.info(f"数据更新成功，更新了 {len(updated_data)} 条记录")
            return "", f"已更新 {len(updated_data)} 条记录"
        except Exception as e:
            error_msg = f"数据更新失败：{str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg, "更新失败"

    def copy_image_to_input_dir(self, src_path: str) -> str:
        """复制图片到输入目录"""
        try:
            src_path = Path(src_path)
            # 使用原始文件名
            dest_path = self.input_figure_dir / src_path.name
            # 如果目标文件已存在，添加数字后缀
            if dest_path.exists():
                base = dest_path.stem
                suffix = dest_path.suffix
                counter = 1
                while dest_path.exists():
                    dest_path = self.input_figure_dir / f"{base}_{counter}{suffix}"
                    counter += 1
            
            # 复制文件
            import shutil
            shutil.copy2(src_path, dest_path)
            logger.info(f"图片已复制到: {dest_path}")
            return str(dest_path)
        except Exception as e:
            logger.error(f"复制图片失败: {str(e)}")
            return str(src_path)  # 如果复制失败，返回原始路径

if __name__ == "__main__":
    try:
        logger.info("启动应用")
        app = DataMakerApp()
        ui = app.create_ui()
        ui.queue(max_size=20, api_open=False).launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_api=False,
            show_error=True,
            inbrowser=True
        )
    except Exception as e:
        logger.error(f"应用启动失败: {str(e)}", exc_info=True)