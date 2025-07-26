from transformers import pipeline
import torch
import json
import os

# Always use GPU if available
device = 0  # Force GPU usage
print("✅ Using GPU:", torch.cuda.get_device_name(0))

translator = pipeline(
    "translation_en_to_hi",
    model="Helsinki-NLP/opus-mt-en-hi",
    device=device
)
print("🚀 Translation pipeline loaded.")

# Only use the first 1,000 sentences for synthetic generation
INPUT_FILE = "en_hi_seed_data.jsonl"
SYNTHETIC_LIMIT = 1000
BATCH_SIZE = 32

if not os.path.isfile(INPUT_FILE):
    print(f"ERROR: File not found: {INPUT_FILE}")
    exit(1)

# ---- Only load the first 1,000
data = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if idx >= SYNTHETIC_LIMIT:
            break
        data.append(json.loads(line))

synthetic_pairs = []

print(f"Translating {len(data)} English sentences (batch size {BATCH_SIZE}) ...")

for i in range(0, len(data), BATCH_SIZE):
    batch = data[i:i + BATCH_SIZE]
    english_batch = [item["source"] for item in batch]
    try:
        translations = translator(english_batch)
        for src, out in zip(english_batch, translations):
            synthetic_pairs.append({
                "source": src,
                "target": out["translation_text"],
                "synthetic": True
            })
    except Exception as e:
        print(f"Error at batch {i}-{i+BATCH_SIZE}: {e}")

    if (i // BATCH_SIZE) % 5 == 0:
        print(f"  Progress: {i + len(batch)}/{len(data)} sentences")

OUTPUT_FILE = "synthetic_en_hi.jsonl"
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for pair in synthetic_pairs:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")

print(f"✅ Done! Translated and saved {len(synthetic_pairs)} synthetic English–Hindi pairs to {OUTPUT_FILE}.")
