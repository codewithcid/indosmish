# Export QLoRA adapter → merged fp16 → GGUF q4_K_M

Run after `qlora_train.py`. This produces the two artifacts the protocol compares:
a merged **fp16** model (FP16 arm) and a **4-bit GGUF** (q4_K_M arm).

## 1. Merge adapter into base (fp16)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "/kaggle/working/qwen-indosmish/adapter"

base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="cpu")
merged = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()
merged.save_pretrained("/kaggle/working/qwen-merged-fp16")
AutoTokenizer.from_pretrained(BASE).save_pretrained("/kaggle/working/qwen-merged-fp16")
```

Download `qwen-merged-fp16/` — this is the **FP16 arm** for the quantization-delta table.

## 2. Convert to GGUF and quantize to q4_K_M

```bash
git clone https://github.com/ggerganov/llama.cpp
pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py qwen-merged-fp16 --outfile qwen-fp16.gguf --outtype f16
# build llama.cpp, then:
./llama.cpp/build/bin/llama-quantize qwen-fp16.gguf qwen2.5-1.5b-q4_k_m.gguf Q4_K_M
```

Download `qwen2.5-1.5b-q4_k_m.gguf` (~1 GB) → put in local `models/`.
This is the **4-bit arm** and the file the Gradio demo + profiler load.

## 3. Produce the headline delta table (locally)

```powershell
# FP16 arm (needs the merged HF dir; run on Kaggle or a bigger machine if 4 GB is tight)
python -m indosmish.models.slm_prompt --hf path\to\qwen-merged-fp16 --shots 0
# 4-bit arm (local, fits easily)
python -m indosmish.models.slm_prompt --gguf models\qwen2.5-1.5b-q4_k_m.gguf --shots 0
python -m indosmish.eval.build_tables
```
