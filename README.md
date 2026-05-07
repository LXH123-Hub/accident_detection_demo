# 交通事故智能检测系统 Demo

基于 **YOLO11s + Focaler-IoU (SIoU)** 的复杂环境交通异常事件自动检测模型可视化演示。

上传一张交通场景图片，系统将自动识别图片中是否存在交通事故，并以检测框可视化标注事故区域。

## 模型性能

| 指标 | 数值 |
|------|------|
| F1 Score | 0.9730 |
| 最优置信度阈值 | 0.01 |
| 推理图片尺寸 | 832×832 |
| 模型架构 | YOLO11s (含 DSConv 自定义模块) |
| 损失函数 | Focaler-IoU (SIoU) |

## 项目结构

```
accident_detection_demo/
├── demo.py              # 主程序 (Gradio Web UI)
├── weights/
│   └── best.pt          # 模型权重 (示例，可替换为自训练权重)
├── ultralytics/         # YOLO 推理框架 (含自定义模块)
├── requirements.txt     # Python 依赖
├── .gitignore
└── README.md
```

## 环境要求

- Python >= 3.9
- NVIDIA GPU + CUDA (推荐，CPU 也可运行但速度较慢)

## 快速开始

### 1. 安装依赖

```bash
# 建议使用 conda 虚拟环境
conda create -n accident_det python=3.11 -y
conda activate accident_det

# 安装 PyTorch (根据你的 CUDA 版本选择)
# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CPU only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 运行 Demo

```bash
# 使用默认权重 (weights/best.pt)
python demo.py

# 指定自定义权重路径
python demo.py --model /path/to/your/best.pt

# 指定端口和生成公共链接
python demo.py --port 8080 --share
```

### 3. 打开浏览器

运行成功后，终端会输出类似:
```
Running on local URL:  http://0.0.0.0:7860
```

浏览器会自动打开，或手动访问 `http://localhost:7860`。

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `weights/best.pt` | 模型权重文件路径 |
| `--conf` | `0.01` | 置信度阈值 (验证集优化值) |
| `--iou` | `0.7` | NMS IoU 阈值 |
| `--imgsz` | `832` | 推理图片尺寸 |
| `--port` | `7860` | Web 服务端口 |
| `--share` | `false` | 是否生成 Gradio 公共分享链接 |

## 使用说明

1. **上传图片**: 点击左侧上传区域，选择一张交通场景图片（支持 jpg/png/bmp 等格式）
2. **自动检测**: 上传后系统自动开始检测，也可点击"开始检测"按钮手动触发
3. **查看结果**:
   - 如检测到事故: 右侧显示带检测框的图片，下方提示"这是事故图片"
   - 如未检测到事故: 显示原图，下方提示"这是非事故图片"
4. **调节阈值**: 拖动置信度滑块可调节检测灵敏度（值越低越敏感）

## 使用自训练权重

你可以替换 `weights/best.pt` 为自己训练的模型权重，只需确保：
- 模型基于本项目提供的 `ultralytics` 框架训练（包含 DSConv 等自定义模块）
- 训练时的类别配置与 demo 一致（单类别: accident）

## 注意事项

- 首次运行时，模型加载可能需要几秒钟
- 如果端口被占用，使用 `--port` 参数指定其他端口
- GPU 推理速度远快于 CPU，建议使用 NVIDIA GPU
- `weights/best.pt` 为示例权重，实际使用时请确保权重文件存在

## License

本项目仅供学术研究与教学使用。
