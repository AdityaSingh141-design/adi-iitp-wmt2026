# Model 2: Contrastive Assamese to English (SFT + DPO)

LoRA adapter for the contrastive Assamese-to-English system submitted to WMT 2026, trained with a preference-optimization stage on top of the SFT model.

## Details

- **Direction**: Assamese to English
- **Base model**: `ai4bharat/indictrans2-indic-en-dist-200M`
- **Training method**: LoRA SFT followed by Direct Preference Optimization (DPO)
- **LoRA config**: rank 8, alpha 16, dropout 0.1, target `q_proj`
- **DPO config**: beta 0.1, learning rate 1e-5, 3 epochs
- **Reward**: 0.4 GEMBA + 0.4 CometKiwi + 0.2 xCOMET
- **WMT 2026 BLEU**: 25.11

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

base_model = "ai4bharat/indictrans2-indic-en-dist-200M"
adapter_path = "models/asm-eng-bestsft-dpo"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(base_model, trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter_path)
```