import json
import random

locations = [
    "bedroom", "bathroom", "kitchen", "living room", "garage", "balcony", "hallway", 
    "office", "pantry", "dining room", "laundry room", "closet", "storage", "entryway"
]

# 各種說明句模板
location_patterns = [
    "{name} locate at ({x},{y})",
    "the {name} is located at ({x},{y})",
    "{name}: ({x},{y})",
    "coordinates of {name} are ({x},{y})",
    "{name} -> ({x},{y})"
]

# 問句模板
question_patterns = [
    "where is the {target}?",
    "which coordinates correspond to the {target}?",
    "what are the (x,y) of {target}?",
    "find {target} coordinates.",
    "tell me the coordinates of {target}.",
    "please locate {target}.",
    "what are the coordinates of {target}?"
]

dataset = []

for _ in range(20000):
    # 隨機選擇地點數量
    num_locs = random.randint(4, 6)
    chosen = random.sample(locations, num_locs)
    
    loc_lines = []
    coord_map = {}
    
    for name in chosen:
        x = random.randint(-10, 10)
        y = random.randint(-10, 10)
        coord_map[name] = (x, y)
        loc_lines.append(random.choice(location_patterns).format(name=name, x=x, y=y))
    
    # 隨機選一個地點來詢問
    target = random.choice(chosen)
    question = random.choice(question_patterns).format(target=target)
    answer = f"The coordinates of the {target} are ({coord_map[target][0]},{coord_map[target][1]})."
    
    # 組合 user/assistant 格式
    user_prompt = "\n".join(loc_lines) + "\n" + question
    assistant_reply = answer

    dataset.append({
        "input": user_prompt,
        "output": assistant_reply
    })

# 輸出 JSONL 檔案
with open("mapping_dataset.jsonl", "w", encoding="utf-8") as f:
    for data in dataset:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")

print("✅ 已生成 mapping_dataset.jsonl，共 20000 筆。")