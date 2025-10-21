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

def generate_improved_mapping_dataset(n_samples=200000, filename="mapping_v2.jsonl"):
    with open(filename, "w", encoding="utf-8") as f:
        for _ in tqdm(range(n_samples), desc="Generating v2 dataset"):
            num_locs = random.randint(3, 10)  # 增加上下文長度
            chosen_locs = random.sample(LOCATIONS, k=num_locs)
            coords = {loc: (random.randint(-50, 50), random.randint(-50, 50)) for loc in chosen_locs}

            # 增加負數座標比例
            for loc in coords:
                if random.random() < 0.7:  # 60% 機率有負數
                    coords[loc] = (coords[loc][0] * random.choice([-1, 1]), 
                                 coords[loc][1] * random.choice([-1, 1]))

            query_idx = random.choices(range(num_locs), weights=[0.05] + [0.15]*(num_locs-1))[0]
            query_loc = chosen_locs[query_idx]
            query_sentence = random.choice(QUERY_TEMPLATES).format(place=query_loc)
            
            random.shuffle(chosen_locs)
            coord_map = {loc: coords[loc] for loc in chosen_locs}
            
            # 改進輸入格式：更明確的結構化提示
            input_lines = []
            for loc in chosen_locs:
                x, y = coord_map[loc]
                input_lines.append(f"{loc}: ({x},{y})")
            input_lines.append(f"QUERY: {query_sentence}")
            input_lines.append("ANSWER FORMAT: (x,y)")

            input_text = "\n".join(input_lines)
            output_text = f"({coord_map[query_loc][0]},{coord_map[query_loc][1]})"

            # 增加格式約束
            sample = {
                "input": input_text,
                "output": output_text
            }
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")

if __name__ == "__main__":
    generate_improved_mapping_dataset()