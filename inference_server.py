#!/usr/bin/env python3
"""
EfficientNet-B3 图像分类推理服务器

用法:
    python classification_server.py --host 0.0.0.0 --port 22335 --device cpu

API端点:
    POST /inference - 图像分类推理
    GET /health - 健康检查
    GET /docs - API文档
"""

import argparse
import base64
import io
import time
import torch
from PIL import Image
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from huggingface_hub import hf_hub_download

from common import EfficientNetClassifier, get_transforms


class ClassificationRequest(BaseModel):
    """分类请求模型"""

    image: str  # Base64编码的图像数据
    image_format: str = "jpeg"  # 图像格式


class ClassificationResponse(BaseModel):
    """分类响应模型"""

    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


class ClassificationInferenceServer:
    """图像分类推理服务器类"""

    def __init__(
        self, device: str = "cpu", model_path: str = None, image_size: int = 300
    ):
        self.device = torch.device(device)
        self.image_size = image_size
        self.model = None
        self.transform = None
        self.model_path = model_path

    def load_model(self):
        """加载EfficientNet-B3分类模型"""
        print("正在加载EfficientNet-B3分类模型...")

        # 确定模型路径
        if self.model_path:
            model_path = self.model_path
            print(f"使用本地模型: {model_path}")
        else:
            # 从Hugging Face下载
            model_path = hf_hub_download(
                repo_id="Curioushi61/BoxAutoLabel",
                filename="box_classification.pth",
                cache_dir="./models",
            )
            print(f"从Hugging Face下载模型到: {model_path}")

        # 创建模型
        self.model = EfficientNetClassifier(num_classes=2, pretrained=False)

        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.to(self.device)
        self.model.eval()

        # 设置预处理变换
        _, self.transform = get_transforms(self.image_size)

        print("模型加载完成")

    def decode_base64_image(self, image_data: str, image_format: str) -> Image.Image:
        """解码Base64图像数据"""
        try:
            # 移除可能的data URL前缀
            if image_data.startswith("data:image/"):
                image_data = image_data.split(",")[1]

            # 解码Base64数据
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            # 转换为RGB模式
            if image.mode != "RGB":
                image = image.convert("RGB")

            return image
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"图像解码失败: {str(e)}")

    def run_inference(self, image: Image.Image) -> Dict[str, Any]:
        """运行模型推理"""
        print("开始预处理图像...")

        # 预处理图像
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        print(f"预处理完成，图像形状: {image_tensor.shape}")

        # 运行推理
        print("开始推理...")
        start_time = time.time()

        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        inference_time = time.time() - start_time
        print(f"推理完成，耗时: {inference_time:.2f}秒")

        # 准备结果
        results = {
            "class_id": predicted_class,
            "confidence": confidence,
            "processing_time": inference_time,
        }

        print("推理处理完成")
        return results


# 全局服务器实例
server = None


def create_app(device: str = "cpu", model_path: str = None, image_size: int = 300):
    """创建FastAPI应用"""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期管理"""
        # 启动时加载模型
        global server
        server = ClassificationInferenceServer(
            device=device, model_path=model_path, image_size=image_size
        )
        server.load_model()
        yield
        # 关闭时清理资源（如果需要）

    app = FastAPI(
        title="图像分类推理服务器",
        description="基于EfficientNet-B3的图像分类HTTP服务",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health_check():
        """健康检查端点"""
        return {"status": "healthy", "message": "图像分类推理服务器运行正常"}

    @app.post("/inference", response_model=ClassificationResponse)
    async def inference_endpoint(request: ClassificationRequest):
        """图像分类推理端点"""
        try:
            # 解码图像
            image = server.decode_base64_image(request.image, request.image_format)

            # 运行推理
            results = server.run_inference(image)

            return ClassificationResponse(
                status="success",
                message=f"推理完成，耗时 {results['processing_time']:.2f} 秒",
                data=results,
            )

        except Exception as e:
            return ClassificationResponse(status="error", message=f"推理失败: {str(e)}")

    return app


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="图像分类推理服务器")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=22335, help="服务器端口")
    parser.add_argument("--device", type=str, default="cpu", help="使用设备 (cpu/cuda)")
    parser.add_argument(
        "--model", type=str, help="模型文件路径 (可选，不提供则从Hugging Face下载)"
    )
    parser.add_argument("--image-size", type=int, default=300, help="输入图像尺寸")
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    print("启动图像分类推理服务器...")
    print(f"主机: {args.host}")
    print(f"端口: {args.port}")
    print(f"设备: {args.device}")
    print(f"图像尺寸: {args.image_size}")
    if args.model:
        print(f"模型路径: {args.model}")
    else:
        print("模型: 从Hugging Face自动下载")
    print(f"API文档: http://{args.host}:{args.port}/docs")

    app_instance = create_app(
        device=args.device, model_path=args.model, image_size=args.image_size
    )

    uvicorn.run(app_instance, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
