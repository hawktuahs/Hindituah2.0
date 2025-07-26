from datasets import load_dataset
import json

# Load BPCC Hindi Devanagari split
dataset = load_dataset("ai4bharat/BPCC", "bpcc-seed-latest", split="hin_Deva")

# Print one entry and the features
print(dataset[0])
print(dataset.features)

# Filter for English–Hindi pairs
en_hi_pairs = [ex for ex in dataset if ex['src_lang'] == 'eng_Latn' and ex['tgt_lang'] == 'hin_Deva']
print(f"Found {len(en_hi_pairs)} English-Hindi pairs")

# Save as .jsonl
with open("en_hi_seed_data.jsonl", "w", encoding="utf-8") as f:
    for item in en_hi_pairs[:10000]:  # or fewer if less than 10,000 found
        f.write(json.dumps({"source": item["src"], "target": item["tgt"]}, ensure_ascii=False) + '\n')

print(f"✅ Saved {min(len(en_hi_pairs),10000)} English-Hindi pairs!")
