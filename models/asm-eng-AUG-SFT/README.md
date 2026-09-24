# Model 1: Primary Assamese to English (SFT only)

LoRA adapter for the primary Assamese-to-English system submitted to WMT 2026.

## Details

- **Direction**: Assamese to English
- **Base model**: `ai4bharat/indictrans2-indic-en-dist-200M`
- **Training method**: LoRA supervised fine-tuning (SFT only)
- **LoRA config**: rank 8, alpha 16, dropout 0.1, target `q_proj`
- **WMT 2026 BLEU**: 24.08

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

base_model = "ai4bharat/indictrans2-indic-en-dist-200M"
adapter_path = "models/asm-eng-AUG-SFT"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(base_model, trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter_path)
```