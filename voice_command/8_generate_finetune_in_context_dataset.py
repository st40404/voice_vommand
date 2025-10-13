import json
import random

# 擴充地點清單 (25 個)
locations = [
    "kitchen", "bedroom", "bathroom", "living room", "garage", "office",
    "garden", "balcony", "dining room", "hallway", "attic", "basement",
    "rooftop", "study room", "storage", "laundry room", "guest room",
    "playroom", "conference room", "server room", "gym", "library",
    "theater", "parking lot", "workshop"
]

# 查詢句模板 (50 種不同問法)
query_templates = [
    "where is {place}?",
    "where is {place} located?",
    "tell me the location of {place}",
    "do you know where {place} is?",
    "can you give me {place}'s coordinates?",
    "what is the location of {place}?",
    "please tell me where {place} is",
    "give me the coordinates of {place}",
    "where exactly is {place}?",
    "show me {place} location",
    "tell me where {place} locate",
    "where can I find {place}?",
    "what are the coordinates for {place}?",
    "please provide location of {place}",
    "where about is {place}?",
    "could you tell me {place}'s position?",
    "how do I locate {place}?",
    "where should I go for {place}?",
    "where can I locate {place}?",
    "find me the coordinates of {place}",
    "where is the {place}?",
    "tell me {place} position",
    "what position is {place}?",
    "point me to {place}",
    "can you locate {place}?",
    "what is {place}'s position?",
    "how do I find {place}?",
    "please tell me the coordinates for {place}",
    "I need the location of {place}",
    "where would I find {place}?",
    "could you show me {place}'s coordinates?",
    "can you tell me the exact spot of {place}?",
    "how far is {place}?",
    "which direction is {place}?",
    "can you give me the map point of {place}?",
    "show me where {place} is on the map",
    "what is the exact location of {place}?",
    "how do I get to {place}?",
    "where can I reach {place}?",
    "what are the map coordinates for {place}?",
    "is {place} on the map?",
    "please locate {place} for me",
    "help me find {place}",
    "where should I search for {place}?",
    "guide me to {place}",
    "what’s the coordinate of {place}?",
    "locate {place} for me",
    "map me to {place}",
    "find {place} location",
    "give me {place}'s map coordinates"
]

def generate_dataset(n_samples=200000, filename="coordinate_dataset.jsonl"):
    with open(filename, "w") as f:
        for i in range(n_samples):
            # 一半自然語言，一半結構化
            if i % 2 == 0:
                # ===== 格式 A: 自然語言查詢 =====
                format_type = random.choice([1, 2])  # 1=已知→查詢, 2=未知→學習

                if format_type == 1:
                    chosen = random.sample(locations, k=random.randint(5, 8))
                    coords = {loc: (random.randint(-50, 50), random.randint(-50, 50)) for loc in chosen}

                    sample = ""
                    for loc, (x, y) in coords.items():
                        sample += f"Input: {loc} locate at ({x},{y})\nOutput: OK, I got it\n"

                    query = random.choice(chosen)
                    query_sentence = random.choice(query_templates).format(place=query)

                    input_text = sample + f"Input: {query_sentence}\nOutput:"
                    output_text = f"({coords[query][0]},{coords[query][1]})"

                else:
                    query = random.choice(locations)
                    x, y = random.randint(-50, 50), random.randint(-50, 50)

                    query_sentence1 = random.choice(query_templates).format(place=query)
                    query_sentence2 = random.choice(query_templates).format(place=query)

                    input_text = (
                        f"Input: {query_sentence1}\n"
                        f"Output: Sorry, I don't know. Please tell me.\n"
                        f"Input: {query} locate at ({x},{y})\n"
                        f"Output: OK, I got it\n"
                        f"Input: {query_sentence2}\n"
                        f"Output:"
                    )
                    output_text = f"({x},{y})"

            else:
                # ===== 格式 B: 結構化查詢 =====
                chosen = random.sample(locations, k=random.randint(3, 6))
                coords = {loc: (random.randint(-50, 50), random.randint(-50, 50)) for loc in chosen}

                sample = ""
                for loc, (x, y) in coords.items():
                    sample += f"Input: set({loc},({x},{y}))\nOutput: OK\n"

                query = random.choice(chosen)
                input_text = sample + f"Input: query({query})\nOutput:"
                output_text = f"({coords[query][0]},{coords[query][1]})"

            # 存成 JSONL
            json.dump({"input": input_text, "output": output_text}, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ Dataset 已生成: {filename} (共 {n_samples} 筆，自然語言 + 結構化混合)")

# 預設生成 20 萬筆資料
generate_dataset(200000, "coordinate_dataset.jsonl")
