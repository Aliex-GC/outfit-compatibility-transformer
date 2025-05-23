import os
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from dataclasses import dataclass
from typing import List
import torch.nn as nn
import numpy as np
# 项目模块导入
from .vectorstore import FAISSVectorStore
from ..models.load import load_model
from ..data import datatypes
from ..data.datasets import polyvore
from ..models.modules.text_encoder import CLIPTextEncoder  # 确保路径正确

POLYVORE_PRECOMPUTED_REC_EMBEDDING_DIR = "{polyvore_dir}/precomputed_rec_embeddings"
# 全局状态容器
class AppState:
    def __init__(self):
        self.my_items = []
        self.polyvore_dataset = None
        self.compatibility_model = None
        self.indexer = None
        self.description=""
    def clear(self):
        self.my_items = []
state = AppState()
app = Flask(__name__)
CORS(app)

# 工具函数
def pil_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def base64_to_pil(base64_str: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(base64_str)))

# API 端点
@app.route('/items', methods=['POST'])
def handle_items():
    if request.method == 'POST':
        data = request.json
        try:
            if not all(k in data for k in ["image", "description", "category"]):
                return jsonify({"error": "Missing required fields"}), 400
            state.clear()
            state.description=data["description"]
            new_item = datatypes.FashionItem(
                id=None,
                image=base64_to_pil(data["image"]),
                
                description=data["description"],
                category=data["category"]
            )
            state.my_items.append(new_item)
            return jsonify({"message": "Item added", "count": len(state.my_items)}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/items/<int:index>', methods=['DELETE'])
def remove_item(index):
    try:
        if 0 <= index < len(state.my_items):
            del state.my_items[index]
            return jsonify({"message": "Item deleted"}), 200
        return jsonify({"error": "Invalid index"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/polyvore', methods=['GET'])
def get_polyvore_page():
    try:
        page = int(request.args.get('page', 1))
        per_page = 12
        start = (page-1)*per_page
        end = start + per_page
        
        return jsonify({
            "items": [
                {
                    "image": pil_to_base64(item.image),
                    "description": item.description,
                    "category": item.category
                }
                for item in state.polyvore_dataset[start:end]
            ]
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/search', methods=['POST'])
def search_complementary():
    if request.method == 'POST':
        try:
            data = request.json
            query = datatypes.FashionComplementaryQuery(
                outfit=state.my_items,
                category=data['target_category']
                # category="bags"
            )
            
            with torch.no_grad():
                embedding = state.compatibility_model.embed_query(
                    query=[query],
                    use_precomputed_embedding=False
                ).cpu().numpy()
                
            # print(embedding.shape)
            results = state.indexer.search(embedding, k=512)[0]
            # print(type(results))
            filtered_results = []
            for score, item_id in results:
                item = state.polyvore_dataset.get_item_by_id(item_id)
                if item.category == query.category:  # query 是之前传入的 FashionComplementaryQuery 对象
                    filtered_results.append((score, item_id))

            # 2. 去重并保留最高分项（原逻辑）
            score_map = {}
            for score, item_id in filtered_results:
                if item_id not in score_map or score > score_map[item_id][0]:
                    score_map[item_id] = (score, item_id)
            
            print(len(filtered_results))
            return jsonify({
                "results": [
                    {
                        "image":pil_to_base64(state.polyvore_dataset.get_item_by_id(item_id).image),
                        "id":item_id,
                        "score":float(score)
                    }
                    for score, item_id in sorted(score_map.values(), key=lambda x: -x[0])  # 按分数降序排列
                    
                ]
            }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

def text_to_embedding(text_query: str, text_encoder: CLIPTextEncoder) -> np.ndarray:
    """使用CLIPTextEncoder处理文本"""
    # 注意：输入需要是List[List[str]]格式，外层List是batch，内层List是序列
    with torch.no_grad():
        embeddings = text_encoder([[text_query]])  # (1, 1, d_embed)
        
     # 2. 初始化投影层（512->128）
        projection = nn.Linear(512, 128, bias=False).to(embeddings.device)
        
        # 3. 降维处理
        projected = projection(embeddings[0, 0])  # (128,)
        
        # 4. 转换为numpy（必须先detach）
        print(projected.detach().cpu().numpy().shape)
        return projected.detach().cpu().numpy()

@app.route('/search1', methods=['POST'])
def find_clip():
    if request.method == 'POST':
        try:
            data = request.json
            text=data["text"]
            target_category=data["target_category"]
            print(text,target_category)
            text_encoder = CLIPTextEncoder(model_name_or_path='models/fashion-clip')
            embedding = text_to_embedding(text,text_encoder)
            embedding = np.expand_dims(embedding, axis=0)
            results = state.indexer.search(embedding, k=256)[0]
            # print(type(results))
            filtered_results = []
            idxx=0

            for score, item_id in results:
                item = state.polyvore_dataset.get_item_by_id(item_id)
                if item.category == target_category:  
                    filtered_results.append((score, item_id))
                    idxx+=1
                if idxx>10:
                    break
            # 2. 去重并保留最高分项（原逻辑）
            score_map = {}
            for score, item_id in filtered_results:
                if item_id not in score_map or score > score_map[item_id][0]:
                    score_map[item_id] = (score, item_id)
            
            print(len(filtered_results))
            return jsonify({
                "results": [
                    {
                        "image":pil_to_base64(state.polyvore_dataset.get_item_by_id(item_id).image),
                        "id":item_id,
                        "score":float(score)
                    }
                    for score, item_id in sorted(score_map.values(), key=lambda x: -x[0])  # 按分数降序排列
                    
                ]
            }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
        
# 初始化函数
def initialize_backend(
    model_type: str = "clip",
    compatibility_checkpoint: str = "checkpoints/complementary_clip_best.pth",
    polyvore_dir: str = "./datasets/polyvore"
):
    """初始化后端服务
    
    Args:
        model_type: 模型类型 (original/clip)
        compatibility_checkpoint: 兼容性模型检查点路径
        polyvore_dir: Polyvore数据集目录
    """
    # 加载数据集
    metadata = polyvore.load_metadata(polyvore_dir)
    state.polyvore_dataset = polyvore.PolyvoreItemDataset(
        polyvore_dir,
        metadata=metadata,
        load_image=True
    )
    
    # 初始化模型
    state.compatibility_model = load_model(
        model_type=model_type,
        checkpoint=compatibility_checkpoint
    ).eval()
    
    
    # 初始化FAISS
    state.indexer = FAISSVectorStore(
        index_name='rec_index',
        d_embed=128,
        faiss_type='IndexFlatIP',
        base_dir=POLYVORE_PRECOMPUTED_REC_EMBEDDING_DIR.format(
            polyvore_dir=polyvore_dir
        ),
    )

def start_server(host: str = "0.0.0.0", port: int = 5000):
    """启动服务"""
    app.run(host=host, port=port,debug=True)

if __name__ == '__main__':
    # 使用示例
    initialize_backend(
        model_type="clip",
        compatibility_checkpoint="checkpoints/compatibillity_clip_best.pth",
        polyvore_dir="./datasets/polyvore"
    )
    start_server()
    
    
    
    
    
    #正面系带人造丝衬衫
    
    
    #python -m src.demo.flask