import json
import jieba

# 加载JSON数据
with open('output.json', 'r', encoding='utf-8') as f:
    clothing_data = json.load(f)
    print(f"加载了 {len(clothing_data)} 个服装数据项。")

def chinese_word_segmentation(text):
    return list(jieba.cut(text))

def search_by_keywords(query, data):
    results = []
    query_words = chinese_word_segmentation(query.lower())  # 提前分词
    
    for item in data:
        # 准备要搜索的文本内容
        text_to_search = f"{item.get('description', '')} {item.get('title', '')} {' '.join(item.get('categories', []))} {' '.join(item.get('related', []))}"
        text_words = chinese_word_segmentation(text_to_search.lower())

        # 精确单词匹配
        match_count = sum(1 for q_word in query_words if q_word in text_words)
        
        if match_count > 0:
            results.append((item, match_count))
    
    # 按匹配度排序
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# 示例使用
user_query = "时尚百搭的短裤"
matches = search_by_keywords(user_query, clothing_data)
print(f"找到 {len(matches)} 个匹配项：")

# 打印前10个匹配结果
for i, (item, score) in enumerate(matches[:10], 1):
    print(f"\n{i}. 匹配度: {score}")
    print(f"商品ID: {item.get('item_id')}")
    print(f"标题: {item.get('title')}")
    print(f"描述: {item.get('description')}")
    print(f"类别: {item.get('semantic_category')}")