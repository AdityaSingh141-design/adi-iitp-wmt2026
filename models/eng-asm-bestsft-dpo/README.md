# Model 3: Primary English to Assamese (SFT + DPO)

LoRA adapter for the primary English-to-Assamese system submitted to WMT 2026, trained with a preference-optimization stage on top of the SFT model.

## Details

- **Direction**: English to Assamese
- **Base model**: `ai4bharat/indictrans2-en-indic-dist-200M`
- **Training method**: LoRA SFT followed by Direct Preference Optimization (DPO)
- **LoRA config**: rank 8, alpha 16, dropout 0.1, target `q_proj`
- **DPO config**: beta 0.1, learning rate 1e-5, 3 epochs
- **Reward**: 0.4 GEMBA + 0.4 CometKiwi + 0.2 xCOMET
- **WMT 2026 BLEU**: 15.57

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

base_model = "ai4bharat/indictrans2-en-indic-dist-200M"
adapter_path = "models/eng-asm-bestsft-dpo"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(base_model, trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter_path)
```