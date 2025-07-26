# LLaMA-2 Hindi Translation Fine-Tuning

This project demonstrates fine-tuning LLaMA-2 3B model for English-Hindi translation using LoRA and synthetic data.

## Project Structure

```
LLMRevamped/
├── fine_tune_llama.py        # Main fine-tuning script
├── evaluate_lora.py          # Model evaluation script
├── load_bpcc.py             # Downloads and processes BPCC dataset
├── generate_synthetic_translations.py  # Generates synthetic translations
├── split_dataset.py          # Splits data into train/val/test sets
├── analyze_translations.py   # Analyzes translation quality
├── requirements.txt          # Project dependencies
└── synthetic_en_hi.jsonl     # Synthetic translation dataset
```

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- CUDA 11.8+ (cu118 wheels)
- Hugging Face account with API token

## Installation

1. Create and activate virtual environment:
```bash
python -m venv llama_env
llama_env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. Load BPCC dataset:
```bash
python load_bpcc.py
```

2. Generate synthetic translations:
```bash
python generate_synthetic_translations.py
```

3. Split data into train/val/test sets:
```bash
python split_dataset.py
```

## Fine-Tuning

1. Fine-tune LLaMA-2 with LoRA:
```bash
python fine_tune_llama.py
```

## Evaluation

1. Evaluate the fine-tuned model:
```bash
python evaluate_lora.py
```

## Configuration

The main configuration parameters are in `fine_tune_llama.py`:
- `model_name`: Base model to use (default: "meta-llama/Llama-3.2-3B-Instruct")
- `BATCH_SIZE`: Training batch size (default: 2)
- `EPOCHS`: Number of training epochs (default: 2)
- `MAX_STEPS`: Maximum training steps (default: 300)
- `GRAD_ACC`: Gradient accumulation steps (default: 4)

## Model Architecture

- Base model: LLaMA-2 3B
- Fine-tuning: LoRA with 4-bit quantization
- Training framework: Hugging Face Transformers + PEFT
- Training strategy: Supervised Fine-Tuning (SFT)

## Evaluation Metrics

The model is evaluated using:
- BLEU score
- chrF score
- Sample translations comparison

## Notes

- Make sure to replace `HF_TOKEN` in `fine_tune_llama.py` with your actual Hugging Face API token
- The model uses 4-bit quantization to reduce memory usage
- Training can be resource-intensive - adjust batch size and gradient accumulation if needed
- The evaluation script requires the `sacrebleu` package for metric calculation

## License

This project is for educational and research purposes only. Please refer to the LLaMA-2 license for usage restrictions.
