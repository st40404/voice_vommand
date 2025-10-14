import json
import random
from tqdm import tqdm

# 可用地點
LOCATIONS = [
    "kitchen", "bedroom", "bathroom", "living room",
    "garage", "office", "dining room", "balcony",
    "study", "laundry", "pantry", "hallway"
]

# 查詢句模板
QUERY_TEMPLATES = [
    "where is {place}?",
    "what is the location of {place}?",
    "tell me {place}'s coordinates",
    "please provide the location of {place}",
    "can you give me {place}'s coordinates?",
    "find the position of {place}",
    "which coordinates correspond to {place}?",
    "what are the coordinates of {place}?",
    "locate {place} for me",
    "show me where {place} is"
]

def generate_mapping_dataset(n_samples=20000, filename="mapping_dataset.jsonl"):
    with open(filename, "w", encoding="utf-8") as f:
        for _ in tqdm(range(n_samples), desc="Generating dataset"):
            # 隨機抽 3~6 個地點
            chosen_locs = random.sample(LOCATIONS, k=random.randint(3, 6))

            # 為每個地點生成隨機座標
            coords = {loc: (random.randint(-10, 10), random.randint(-10, 10)) for loc in chosen_locs}

            # 隨機挑一個要查詢的地點
            query_loc = random.choice(chosen_locs)
            query_sentence = random.choice(QUERY_TEMPLATES).format(place=query_loc)

            # 組合 input
            input_lines = []
            for loc, (x, y) in coords.items():
                input_lines.append(f"{loc} locate at ({x},{y})")
            input_lines.append(query_sentence)

            input_text = "\n".join(input_lines)
            output_text = f"({coords[query_loc][0]},{coords[query_loc][1]})"

            json.dump({"input": input_text, "output": output_text}, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ 已生成 {n_samples} 筆 mapping dataset：{filename}")

if __name__ == "__main__":
    generate_mapping_dataset()