import requests
import base64
from gradio_client import Client

from src.data import datatypes
def call_compute_score():
    """调用兼容性分数计算API"""
    url = "http://localhost:7860/compute_score"
    # ==== 图片转Base64示例 ====
    def image_to_base64(file_path):
        with open(file_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    
    # 使用示例（替换为你的图片路径）
    image_path1 = "datasets/polyvore/images/117427809.jpg"
    image_path2 = "datasets/polyvore/images/197823931.jpg"
    
    items = [
        {
            "image": image_to_base64(image_path1),
            "description": "Tie front rayon shirt",
            "category": "tops"
        },
        {
            "image": image_to_base64(image_path2),
            "description": "peacock feather appliques backpack",
            "category": "bags"
        }
    ]
    response = requests.post(url, json={"items": items})
    print("API返回内容:", response.json())
    print(f"兼容性分数: {response.json()['score']}")

def call_search_complementary():
    def image_to_base64(file_path):
        with open(file_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    """调用互补物品搜索API"""
    url = "http://localhost:7860/search_items"
    # 互补物品查询示例
    query_image_path = "path/to/red_handbag.jpg"
    query = {
        "image": image_to_base64(query_image_path),
        "description": "Red Handbag",
        "category": "bags"
    }
    response = requests.post(url, json=query)
    results = response.json()
    for idx, item in enumerate(results):
        print(f"推荐物品{idx+1}:")
        print(f"  描述: {item['description']}")
        print(f"  分数: {item['score']:.2f}")
        # 保存图片到本地示例（可选）
        # with open(f"result_{idx}.jpg", "wb") as f:
        #     f.write(base64.b64decode(item["image"]))

if __name__ == "__main__":
    # print("请选择要测试的功能：")
    # print("1. 计算兼容性分数")
    # print("2. 搜索互补物品")
    # choice = input("输入选项编号 > ")
    
    # if choice == '1':
    #     call_compute_score()
    # elif choice == '2':
    #     call_search_complementary()
    # else:
    #     print("无效输入")
    
    def image_to_base64(file_path):
            with open(file_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')

    client = Client("http://127.0.0.1:5000")
    from PIL import Image

    # 读取图像文件
    image_path1 = "datasets/polyvore/images/117427809.jpg"
    image_path2 = "datasets/polyvore/images/197823931.jpg"

    # 使用 Pillow 打开图像文件
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)
    state_my_items=[]
    state_my_items.append(
                    datatypes.FashionItem(
                        id=None,
                        image=image1, 
                        description="Tie front rayon shirt",
                        category="tops",
                    )
                )
    
    # state_my_items.append(
    #                 datatypes.FashionItem(
    #                         id=None,
    #                         image=image2, 
    #                         description="peacock feather appliques backpack",
    #                         category="bags",
    #                         )
    #                     )
    # items = [
    #     {
    #         "image": image_path1,
    #         "description": "Tie front rayon shirt",
    #         "category": "tops"
    #     },
    #     {
    #         "image": image_path2,
    #         "description": "peacock feather appliques backpack",
    #         "category": "bags"
    #     }
    # ]
    result = client.predict( 
        # 使用示例（替换为你的图片路径）
        state_my_items=state_my_items,
        api_name="/search"
    )
    print(result)