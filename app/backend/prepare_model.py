"""Prepare lossless external FP32 weights during the hosting build."""
from pathlib import Path
import onnx

root = Path(__file__).resolve().parent / "model" / "onnx"
source = root / "model.onnx"
target = root / "model.external.onnx"
weights = root / "model.weights.bin"
model = onnx.load(str(source))
# onnx.save_model appends to an existing external file; remove only our
# generated build artifact before regenerating it.
weights.unlink(missing_ok=True)
onnx.save_model(model, str(target), save_as_external_data=True,
                all_tensors_to_one_file=True, location=weights.name,
                size_threshold=1024)
print(f"Prepared {target.name} ({target.stat().st_size:,} bytes) with external FP32 weights")
