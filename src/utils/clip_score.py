import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# 下载并加载 CLIP 模型和处理器
clip_model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(clip_model_name)
processor = CLIPProcessor.from_pretrained(clip_model_name)

# 输入文本和图像
text = "A person wearing a stylish jacket and jeans"
image_path = "path_to_your_image.jpg"

# 加载图像
image = Image.open(image_path)

# 处理文本和图像
inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)

# 获取图像和文本的特征向量
with torch.no_grad():
    text_features = model.get_text_features(**inputs)
    image_features = model.get_image_features(**inputs)

# L2 归一化
text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

# 计算余弦相似度
cosine_similarity = torch.nn.functional.cosine_similarity(text_features, image_features)

# 打印 CLIPScore
print(f"CLIPScore: {cosine_similarity.item()}")
