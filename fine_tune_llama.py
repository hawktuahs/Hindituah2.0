from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
import torch, json

# ====== CONFIGURATION ======
model_name = "meta-llama/Llama-3.2-3B-Instruct"  # or "meta-llama/Llama-3.2-3B"
HF_TOKEN = "YOUR_HF_TOKEN"  # <-- REPLACE with your Hugging Face token

BATCH_SIZE = 2          # reduce to 1 if OOM errors
VAL_BATCH_SIZE = 2
EPOCHS = 2
MAX_STEPS = 300
GRAD_ACC = 4
MAX_SEQ_LENGTH = 256

# ====== LOAD MODEL & TOKENIZER =======
bnb_config = dict(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

print("Loading model and tokenizer (this might take a while the first time)...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    trust_remote_code=True,
    token=HF_TOKEN,
    **bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# ====== PREP LoRA ======
lora_config = LoraConfig(
    r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# ====== PREP DATA ======
def load_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    data = [{"text": f"Translate the following English sentence to Hindi:\nEnglish: {d['source']}\nHindi: {d['target']}"} for d in data]
    return Dataset.from_list(data)

train_data = load_data("train_synthetic_en_hi.jsonl")
val_data = load_data("val_synthetic_en_hi.jsonl")

# ====== TRAINING ARGS ======
training_args = TrainingArguments(
    output_dir="llama2-finetuned-hindi",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=VAL_BATCH_SIZE,
    num_train_epochs=EPOCHS,
    max_steps=MAX_STEPS,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=2e-4,
    save_steps=50,
    logging_steps=10,
    report_to="tensorboard",
    save_total_limit=2,
    fp16=True,
    optim="paged_adamw_8bit",
    warmup_ratio=0.05,
    group_by_length=True,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    evaluation_strategy="steps",  # <--- removed!
    eval_steps=50,               # <--- optional, remove for full compatibility
)

# ====== TRAINER ======
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    peft_config=lora_config
)

trainer.train()
trainer.save_model()
print("✅ LoRA fine-tuning complete! Model saved in llama2-finetuned-hindi")
