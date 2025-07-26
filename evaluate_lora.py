from transformers import AutoModelForCausalLM, AutoTokenizer
from sacrebleu import corpus_bleu, corpus_chrf
from tqdm import tqdm
import torch, json

# ----- CONFIG ---------
model_dir = "llama2-finetuned-hindi"
test_file = "test_synthetic_en_hi.jsonl"
device = "cuda" if torch.cuda.is_available() else "cpu"
max_new_tokens = 64

# ----- LOAD MODEL -----
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# ----- LOAD TEST DATA -----
with open(test_file, "r", encoding="utf-8") as f:
    test_data = [json.loads(l) for l in f]

# ---- REDUCE TEST SIZE for Fast Debugging ----
TEST_SIZE = 25    # change to smaller or larger as needed
test_data = test_data[:TEST_SIZE]

# ----- RUN INFERENCE & COLLECT OUTPUTS -----
preds = []
refs = []
for entry in tqdm(test_data, desc="Translating"):
    prompt = f"Translate the following English sentence to Hindi:\nEnglish: {entry['source']}\nHindi:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # ensure tensor is on the right device

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False   # disables sampling; no temp/top_p, runs fast/greedy
        )

    pred = tokenizer.decode(output[0], skip_special_tokens=True)
    pred = pred.replace(prompt, "").strip()
    preds.append(pred)
    refs.append(entry['target'])

bleu = corpus_bleu(preds, [refs])
chrf = corpus_chrf(preds, [refs])

print(f"\nEvaluation results on test set ({len(test_data)} examples):")
print(f"BLEU score:  {bleu.score:.2f}")
print(f"chrF score:  {chrf.score:.2f}")
print("\nSample prediction (first 3):")
for i in range(min(3, len(test_data))):
    print(f"EN: {test_data[i]['source']}\nPRED: {preds[i]}\nREF: {refs[i]}\n")
