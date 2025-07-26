import json
import random

input_file = "synthetic_en_hi.jsonl"
with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(l) for l in f]
random.shuffle(data)

n = len(data)
train = data[:int(0.8 * n)]
val = data[int(0.8 * n):int(0.9 * n)]
test = data[int(0.9 * n):]

for split, arr in zip(['train', 'val', 'test'], [train, val, test]):
    with open(f"{split}_synthetic_en_hi.jsonl", "w", encoding="utf-8") as f:
        for item in arr:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Split complete!")
