---
title: IndoSmish Smishing Detector
emoji: 📱
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# IndoSmish — Code-Mixed Malayalam–English Smishing Detector

Demo for the IEEE Access 2026 survey benchmark. Tri-class ham/spam/smishing detection.

## Deploy to Hugging Face Spaces

1. Create a new Space (SDK: Gradio, hardware: free CPU basic).
2. Push `app.py`, this `README.md`, and `requirements.txt` (below) to the Space repo.
3. Set Variables in Space settings:
   - `MODEL_KIND` = `encoder` (loads a fine-tuned MuRIL from the Hub) or `gguf`.
   - `MODEL_PATH` = your HF model repo id (e.g. `codewithcid/indosmish-muril`) or gguf path.
4. For the `encoder` path, push your fine-tuned model to a HF model repo first
   (`huggingface-cli upload`). For `gguf`, add the .gguf via Git LFS.

## requirements.txt for the Space

```
gradio>=4.44
transformers>=4.44
torch
# add llama-cpp-python only if MODEL_KIND=gguf
```
