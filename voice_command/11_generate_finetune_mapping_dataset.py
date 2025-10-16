import json
import random
from tqdm import tqdm

LOCATIONS = [
    "kitchen", "bedroom", "bathroom", "living room", "garage", "office",
    "garden", "balcony", "dining room", "hallway", "attic", "basement",
    "rooftop", "study room", "storage", "laundry room", "guest room",
    "playroom", "conference room", "server room", "gym", "library",
    "theater", "parking lot", "workshop"
]

QUERY_TEMPLATES = [
    "where is {place}?",                    # 最自然、直觀
    "what is the location of {place}?",     # 正式、明確
    "coordinates of {place}?"               # 技術性、簡潔
]

def generate_improved_mapping_dataset(n_samples=50000, filename="mapping_improved.jsonl"):
    with open(filename, "w", encoding="utf-8") as f:
        for _ in tqdm(range(n_samples), desc="Generating improved dataset"):
            # 隨機地點數量：3~8 個，增加長上下文訓練
            num_locs = random.randint(3, 8)
            chosen_locs = random.sample(LOCATIONS, k=num_locs)
            
            # 隨機座標，避免模式
            coords = {loc: (random.randint(-20, 20), random.randint(-20, 20)) for loc in chosen_locs}
            
            # 關鍵改進：隨機選擇查詢位置（特別強調中後位置）
            # 增加中後位置查詢的概率
            position_weights = [0.1 if i < num_locs//2 else 0.2 for i in range(num_locs)]
            query_idx = random.choices(range(num_locs), weights=position_weights)[0]
            query_loc = chosen_locs[query_idx]
            query_sentence = random.choice(QUERY_TEMPLATES).format(place=query_loc)
            
            # 隨機打亂順序，避免位置偏置
            random.shuffle(chosen_locs)
            coord_map = {loc: coords[loc] for loc in chosen_locs}
            
            # 組合 input：打亂後的順序
            input_lines = [f"{loc} locate at ({coord_map[loc][0]},{coord_map[loc][1]})" 
                          for loc in chosen_locs]
            input_lines.append(query_sentence)
            
            input_text = "\n".join(input_lines)
            output_text = f"({coord_map[query_loc][0]},{coord_map[query_loc][1]})"
            
            json.dump({"input": input_text, "output": output_text}, f, ensure_ascii=False)
            f.write("\n")
    
    print(f"✅ 已生成改進版 {n_samples} 筆 dataset：{filename}")

if __name__ == "__main__":
    generate_improved_mapping_dataset()