# 05 — Deployment Plan

## AutoCompress — Inference & Serving Plan

---

## 1. Deployment Philosophy

AutoCompress is **not a serving system** — it is a compression pipeline whose output is a deployable artifact. This plan covers:
1. How the best-candidate models are exported and validated
2. How latency is benchmarked against Android / iOS targets
3. What a downstream mobile team needs to take the `.tflite` artifact and ship it

---

## 2. Export Pipeline

### 2.1 Three-Format Export Chain

```
Compressed nn.Module (PyTorch)
       │
       ├──► torch.jit.script() ────────────► model.pt         (TorchScript)
       │
       ├──► torch.onnx.export() ───────────► model.onnx       (ONNX)
       │           │
       │           └──► onnx-tf ──► tf.saved_model
       │                                     │
       │                           tf.lite.TFLiteConverter
       │                                     │
       └────────────────────────────────────► model.tflite     (TFLite / CoreML)
```

### 2.2 Export Code Stubs

**TorchScript:**
```python
scripted = torch.jit.script(model.eval())
scripted.save("model_TS.pt")
```

**ONNX:**
```python
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy, "model.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}},
    do_constant_folding=True,
)
# Verify export
import onnx, onnxruntime as ort
onnx.checker.check_model("model.onnx")
sess = ort.InferenceSession("model.onnx")
```

**TFLite (via ONNX):**
```python
# Step 1: ONNX → TF SavedModel
from onnx_tf.backend import prepare
import onnx
onnx_model = onnx.load("model.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("model_tf_savedmodel")

# Step 2: TF SavedModel → TFLite (INT8 quantized)
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model("model_tf_savedmodel")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = calibration_dataset_gen()
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
```

**CoreML (iOS — optional):**
```python
import coremltools as ct
mlmodel = ct.convert("model.onnx", source="onnx",
                      inputs=[ct.TensorType(shape=(1, 3, 224, 224))])
mlmodel.save("model.mlpackage")
```

---

## 3. Operator Compatibility Checker

After each export, `compat.py` runs a static analysis pass:

```python
def check_tflite_compat(onnx_path: str) -> CompatReport:
    """Check ONNX ops against TFLite supported op set."""
    model = onnx.load(onnx_path)
    unsupported = []
    for node in model.graph.node:
        if node.op_type not in TFLITE_SUPPORTED_OPS:
            unsupported.append(node.op_type)
    return CompatReport(unsupported=unsupported,
                        is_clean=len(unsupported) == 0)
```

Known problem ops for common CNN layers:

| Op | Status | Workaround |
|---|---|---|
| `BatchNormalization` | ✅ Supported | None |
| `GlobalAveragePool` | ✅ Supported | None |
| `Conv` | ✅ Supported | None |
| `Relu6` (MobileNet) | ✅ Supported | None |
| `Upsample` / `Resize` | ⚠️ Partial | Use `opset_version≥13` |
| Custom ops (e.g., SVD layers) | ❌ May fail | Fall back to ONNX Runtime only |

---

## 4. Latency Benchmarking on Android

### 4.1 ADB Benchmarking (Android Debug Bridge)

```bash
# Push TFLite model and benchmark binary to Android device
adb push model.tflite /data/local/tmp/
adb push benchmark_model /data/local/tmp/
adb shell chmod +x /data/local/tmp/benchmark_model

# Run TFLite Benchmark Tool
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/model.tflite \
  --num_threads=4 \
  --num_runs=50 \
  --warmup_runs=5 \
  --use_nnapi=false
```

Captured metrics:
- `avg_inference_timeus` — average latency in microseconds
- `std_inference_timeus` — standard deviation
- `memory_footprint_mb` — peak RAM usage on device

### 4.2 Target Devices for Benchmark

| Device | SoC | RAM | OS | Priority |
|---|---|---|---|---|
| Pixel 6a (test device) | Google Tensor G1 | 6 GB | Android 14 | **Primary** |
| Samsung Galaxy A32 | MediaTek Helio G80 | 4 GB | Android 13 | Secondary (mid-range) |
| Any Android emulator | x86 ABI | N/A | Android 14 | Dev fallback (not representative) |
| iOS (iPhone 12+) | Apple A14 | 4 GB | iOS 17 | CoreML — stretch goal |

> [!NOTE]
> If no physical Android device is available, ONNX Runtime on laptop CPU is used as a proxy. The plan documents this limitation in the benchmark report automatically.

---

## 5. ONNX Runtime Serving (CPU Baseline)

For desktop / embedded Linux targets:

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("model.onnx",
        providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Inference
img = preprocess(raw_image)  # shape: (1, 3, 224, 224)
logits = sess.run([output_name], {input_name: img})[0]
predicted_class = np.argmax(logits)
```

ONNX Runtime provides the most portable CPU inference path and is used for the desktop latency benchmark in the Jupyter report.

---

## 6. Hardware Requirements Summary

### Development Environment

| Component | Minimum | Recommended |
|---|---|---|
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| RAM | 16 GB | 32 GB |
| GPU | NVIDIA GTX 1060 6GB | NVIDIA RTX 3060+ 12GB |
| Storage | 200 GB SSD | 500 GB SSD (for ImageNet) |
| OS | Ubuntu 22.04 / Windows 11 | Ubuntu 22.04 |
| Python | 3.10+ | 3.11 |
| CUDA | 11.8+ | 12.x |

### Mobile Benchmark Device

| Component | Spec |
|---|---|
| Android | 9.0+ (API 28+) |
| ADB-enabled | Developer options unlocked |
| RAM | ≥ 3 GB for INT8 ResNet-50 |

### Cloud (Optional for full ImageNet runs)

| Platform | Config | Est. Cost |
|---|---|---|
| Google Colab Pro | T4 GPU, 25 GB RAM | ~$10/mo |
| AWS EC2 p3.2xlarge | NVIDIA V100, 61 GB RAM | ~$3/hr |
| Kaggle Notebooks | T4 GPU, free tier | Free (30 hrs/week) |

---

## 7. Output Artifacts per Run

```
results/
├── baseline_stats.json              # Baseline model metrics
├── candidate_results.csv            # All 14+ candidates, all metrics
├── best/
│   ├── model_card.json              # Best candidate summary
│   ├── best_model_TS.pt             # TorchScript export
│   ├── best_model.onnx              # ONNX export
│   └── best_model.tflite            # TFLite export
├── all_candidates/
│   ├── quant/
│   │   ├── model.pt
│   │   ├── model.onnx
│   │   └── model.tflite
│   ├── prune_quant/
│   │   └── ...
│   └── ...
└── benchmark_report.ipynb           # Interactive Jupyter report
```

### `model_card.json` schema

```json
{
  "technique_stack": ["prune", "quant"],
  "baseline_model": "resnet50",
  "dataset": "imagenet",
  "accuracy_retention_top1": 0.983,
  "compression_ratio": 4.7,
  "latency_multiplier_cpu": 1.21,
  "latency_ms_android": 48.3,
  "export_formats": ["torchscript", "onnx", "tflite"],
  "tflite_int8": true,
  "input_shape": [1, 3, 224, 224],
  "imagenet_mean": [0.485, 0.456, 0.406],
  "imagenet_std":  [0.229, 0.224, 0.225],
  "quant_scale": 0.003921568,
  "quant_zero_point": -128,
  "generated_at": "2026-08-17T02:00:00Z"
}
```
