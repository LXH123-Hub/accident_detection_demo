#!/usr/bin/env python3
"""交通事故检测 Demo — Anthropic 风格可视化界面。

基于 YOLO11s + Focaler-IoU (V1) 的复杂环境交通异常事件检测模型。
上传一张交通场景图片，系统将自动识别是否存在交通事故并可视化检测结果。

使用方式:
    python demo.py --model weights/best.pt
    python demo.py --model weights/best.pt --conf 0.01 --imgsz 832 --port 7860
    python demo.py --model weights/best.pt --share
"""

from __future__ import annotations

import argparse
import sys
from html import escape
from pathlib import Path

# 将本项目目录中的 ultralytics 包加入 sys.path（包含自定义模块 DSC3k2 等）
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import gradio as gr
except ImportError:
    print("Gradio 未安装。请先运行: pip install gradio")
    sys.exit(1)

import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Anthropic 风格 CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* 全局背景与字体 */
.gradio-container {
    background: #FAF9F6 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    max-width: 1000px !important;
    margin: 0 auto !important;
}

/* 隐藏默认 footer */
footer { display: none !important; }

/* 卡片容器 */
.gr-panel, .gr-box, .gr-form {
    background: #FFFFFF !important;
    border: 1px solid #E8E3DB !important;
    border-radius: 16px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.04) !important;
}

/* 图片容器圆角 */
.gr-image, .image-container {
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* 主按钮 */
.gr-button.primary {
    background: #D97706 !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 10px 24px !important;
    transition: background 0.2s ease !important;
}
.gr-button.primary:hover {
    background: #B45309 !important;
}

/* 滑块样式 */
input[type="range"] {
    accent-color: #D97706 !important;
}

/* 标签文字 */
.gr-input-label, label {
    font-weight: 500 !important;
    color: #374151 !important;
}
"""

# ---------------------------------------------------------------------------
# Header / Footer HTML
# ---------------------------------------------------------------------------
HEADER_HTML = """
<div style="text-align: center; padding: 32px 16px 24px;">
    <div style="
        display: inline-flex; align-items: center; justify-content: center;
        width: 56px; height: 56px; border-radius: 16px;
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        margin-bottom: 16px;
    ">
        <span style="font-size: 28px; line-height: 1;">&#128663;</span>
    </div>
    <h1 style="
        font-size: 1.75rem; font-weight: 700; color: #1F2937;
        margin: 0 0 8px 0; letter-spacing: -0.02em;
    ">交通事故智能检测系统</h1>
    <p style="
        font-size: 0.95rem; color: #6B7280; max-width: 520px;
        margin: 0 auto; line-height: 1.6;
    ">基于 YOLO11s + Focaler-IoU 的复杂环境交通异常事件检测模型。<br>
    上传一张交通场景图片，系统将自动识别是否存在交通事故。</p>
</div>
"""

PLACEHOLDER_HTML = """
<div style="text-align: center; padding: 20px; color: #9CA3AF; font-size: 0.9rem;">
    等待上传图片...
</div>
"""


def build_footer_html(model_path: str, device_info: str) -> str:
    """构建页面底部信息栏。"""
    return f"""
    <div style="
        text-align: center; padding: 16px; margin-top: 8px;
        color: #9CA3AF; font-size: 0.8rem; line-height: 1.6;
    ">
        <span>模型: V1 + rotate_only 增强 (Focaler-IoU)</span>
        <span style="margin: 0 8px;">·</span>
        <span>架构: YOLO11s</span>
        <span style="margin: 0 8px;">·</span>
        <span>设备: {device_info}</span>
        <br>
        <span style="color: #D1D5DB;">权重: {Path(model_path).name}</span>
    </div>
    """


# ---------------------------------------------------------------------------
# 结果 HTML 模板
# ---------------------------------------------------------------------------
def build_status_html(is_accident: bool, num_boxes: int, max_conf: float) -> str:
    """构建状态徽章和检测详情 HTML。"""
    if is_accident:
        badge_style = (
            "background: #FEF2F2; color: #DC2626; border: 1px solid #FCA5A5;"
        )
        badge_icon = "&#9888;"
        badge_text = "这是事故图片"
        detail_text = f"检测到 {num_boxes} 个事故区域，最高置信度 {max_conf:.1%}"
    else:
        badge_style = (
            "background: #F0FDF4; color: #16A34A; border: 1px solid #86EFAC;"
        )
        badge_icon = "&#10003;"
        badge_text = "这是非事故图片"
        detail_text = "图片中未发现交通事故迹象"

    return f"""
    <div style="text-align: center; padding: 16px;">
        <div style="
            display: inline-flex; align-items: center; gap: 8px;
            padding: 10px 24px; border-radius: 999px;
            font-size: 1.05rem; font-weight: 600;
            {badge_style}
        ">
            <span style="font-size: 1.15rem;">{badge_icon}</span>
            {badge_text}
        </div>
        <p style="
            margin-top: 10px; color: #6B7280;
            font-size: 0.88rem; line-height: 1.5;
        ">{detail_text}</p>
    </div>
    """


# ---------------------------------------------------------------------------
# 推理函数
# ---------------------------------------------------------------------------
def create_predict_fn(model: YOLO, iou_threshold: float, imgsz: int, device: str):
    """创建推理闭包，捕获模型实例和固定参数。"""

    def predict(image: Image.Image | None, conf_threshold: float, progress=gr.Progress(track_tqdm=False)):
        if image is None:
            return None, PLACEHOLDER_HTML

        progress(0.1, desc="正在预处理图片...")

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        progress(0.3, desc="正在运行模型推理...")

        try:
            results = model.predict(
                source=image,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )

            progress(0.7, desc="正在生成可视化结果...")

            result = results[0]
            num_boxes = 0 if result.boxes is None else len(result.boxes)

            if num_boxes > 0:
                max_conf = float(result.boxes.conf.max().item())
                output_image = result.plot(
                    pil=True,
                    line_width=2,
                    conf=True,
                    labels=True,
                    boxes=True,
                )
            else:
                max_conf = 0.0
                output_image = image

        except Exception as e:
            error_html = f"""
            <div style="text-align: center; padding: 16px;">
                <div style="
                    display: inline-flex; align-items: center; gap: 8px;
                    padding: 10px 24px; border-radius: 999px;
                    font-size: 1.05rem; font-weight: 600;
                    background: #FEF2F2; color: #DC2626; border: 1px solid #FCA5A5;
                ">推理出错</div>
                <p style="margin-top: 10px; color: #6B7280; font-size: 0.88rem;">{escape(str(e))}</p>
            </div>
            """
            return image, error_html

        progress(1.0, desc="完成")

        status_html = build_status_html(
            is_accident=num_boxes > 0,
            num_boxes=num_boxes,
            max_conf=max_conf,
        )

        return output_image, status_html

    return predict


# ---------------------------------------------------------------------------
# UI 构建
# ---------------------------------------------------------------------------
def build_demo(model: YOLO, args: argparse.Namespace, device: str) -> gr.Blocks:
    """构建 Gradio 界面。"""
    predict_fn = create_predict_fn(model, args.iou, args.imgsz, device)
    device_info = f"CUDA ({torch.cuda.get_device_name(0)})" if device != "cpu" else "CPU"

    theme = gr.themes.Base(
        primary_hue="amber",
        secondary_hue="stone",
        neutral_hue="stone",
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(css=CUSTOM_CSS, theme=theme, title="交通事故检测 Demo") as demo:
        # Header
        gr.HTML(HEADER_HTML)

        # 主体：左右两栏
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="上传图片",
                    type="pil",
                    height=420,
                    sources=["upload", "clipboard"],
                )
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="检测结果",
                    type="pil",
                    height=420,
                    interactive=False,
                )

        # 状态徽章
        status_output = gr.HTML(value=PLACEHOLDER_HTML)

        # 控制区域
        with gr.Row():
            with gr.Column(scale=3):
                conf_slider = gr.Slider(
                    minimum=0.01,
                    maximum=0.50,
                    value=args.conf,
                    step=0.01,
                    label="置信度阈值",
                    info="推荐值 0.01（验证集优化阈值），调高则更严格、调低则更敏感",
                )
            with gr.Column(scale=1, min_width=140):
                detect_btn = gr.Button(
                    "开始检测",
                    variant="primary",
                    size="lg",
                )

        # Footer
        gr.HTML(build_footer_html(args.model, device_info))

        # 事件绑定
        outputs = [output_image, status_output]

        detect_btn.click(
            fn=predict_fn,
            inputs=[input_image, conf_slider],
            outputs=outputs,
            show_progress="full",
        )

        input_image.change(
            fn=predict_fn,
            inputs=[input_image, conf_slider],
            outputs=outputs,
            show_progress="full",
        )

        conf_slider.release(
            fn=predict_fn,
            inputs=[input_image, conf_slider],
            outputs=outputs,
            show_progress="full",
        )

    return demo


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="交通事故检测 Demo — 基于 YOLO11s + Focaler-IoU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python demo.py --model weights/best.pt
  python demo.py --model weights/best.pt --conf 0.01 --imgsz 832 --port 7860
  python demo.py --model weights/best.pt --share
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="weights/best.pt",
        help="模型权重路径 (默认 weights/best.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.01,
        help="默认置信度阈值 (默认 0.01)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU 阈值 (默认 0.7)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=832,
        help="推理图片尺寸 (默认 832)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="服务端口 (默认 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="生成公共分享链接",
    )
    return parser.parse_args()


def detect_device() -> str:
    """自动检测推理设备。"""
    if torch.cuda.is_available():
        device = "0"
        print(f"[设备] 检测到 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[设备] 未检测到 GPU，使用 CPU 推理")
    return device


def main() -> None:
    """启动交通事故检测 Demo。"""
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[错误] 模型文件不存在: {model_path}")
        print("请检查 --model 参数指定的路径是否正确。")
        sys.exit(1)

    device = detect_device()

    print(f"[加载] 正在加载模型: {model_path}")
    model = YOLO(str(model_path))
    print("[加载] 模型加载完成")

    demo = build_demo(model, args, device)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
