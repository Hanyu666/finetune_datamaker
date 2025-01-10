# 数据集创建工具

一个基于 Gradio 的数据集创建工具，用于生成用于模型微调的数据集。该工具支持文本和图像数据的处理，并提供了友好的用户界面。

请注意：本工具完全由cursor编写开发，耗时5小时，存在反复调整bug的过程，难免仍存在bug，请谨慎使用。

## 功能特点

- 支持文本数据处理和图像数据处理
- 提供API设置界面，支持自定义API配置
- 文本处理支持自定义提示词
- 图像处理支持批量生成描述
- 支持数据集的保存和增量更新
- 提供实时处理状态显示
- 支持处理结果的交互式编辑

## 安装要求

- Python 3.8+
- 依赖包：见 requirements.txt

## 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/Hanyu666/finetune_datamaker
cd finetune_datamaker-name
```

2. 创建并激活虚拟环境（可选但推荐）：
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 运行应用：
```bash
python main.py
```

2. 在浏览器中打开显示的本地URL（默认为 http://127.0.0.1:7860）

3. 在API设置标签页中配置您的API密钥和其他设置

4. 根据需要选择文本处理或图像处理标签页进行操作

## 目录结构

```
.
├── main.py              # 主程序
├── modules/             # 模块目录
│   ├── api_config.py    # API配置模块
│   ├── text_processor.py # 文本处理模块
│   └── image_processor.py # 图像处理模块
├── input/             # 输入数据目录
├── output/             # 输出目录
│   └── input_figure_dir/ # 图片存储目录
├── requirements.txt    # 依赖列表
└── README.md          # 项目说明文档
```

## 注意事项

- 请确保在使用前配置正确的API密钥
- 文本文件应使用UTF-8编码
- 图片处理支持常见的图片格式（jpg, png, etc.）
- 处理结果会自动保存在output目录下

## License

MIT License 