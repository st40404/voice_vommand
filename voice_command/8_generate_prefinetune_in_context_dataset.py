import json
import random

# 定義固定地點
locations = [
    "kitchen", "bedroom", "bathroom", "living room",
    "garage", "office", "dining room", "balcony",
    "study", "laundry", "pantry", "hallway"
]

# 查詢模板 (統一格式，先少量)
query_templates = [
    "where is {place}?",
    "what is the location of {place}?",
    "tell me {place}'s coordinates",
    "please provide the location of {place}",
    "can you give me {place}'s coordinates?",
]

def generate_small_dataset(n_samples=500, filename="small_coordinate_dataset.jsonl"):
    with open(filename, "w") as f:
        for _ in range(n_samples):
            # 隨機選 3~5 個地點
            chosen_locs = random.sample(locations, k=random.randint(3,5))
            coords = {loc: (random.randint(1,10), random.randint(1,10)) for loc in chosen_locs}

            # 準備 Input 文本
            input_text = ""
            for loc, (x,y) in coords.items():
                input_text += f"Input: {loc} locate at ({x},{y})\nOutput: OK, I got it\n"

            # 隨機選一個地點做查詢
            query_loc = random.choice(chosen_locs)
            query_sentence = random.choice(query_templates).format(place=query_loc)
            input_text += f"Input: {query_sentence}\nOutput:"

            # Output 就是座標
            output_text = f"({coords[query_loc][0]},{coords[query_loc][1]})"

            # 寫入 JSONL
            json.dump({"input": input_text, "output": output_text}, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ 小型 dataset 已生成: {filename}")

# 生成 500 筆小型 dataset
generate_small_dataset(500)