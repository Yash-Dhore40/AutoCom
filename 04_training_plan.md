# 04 — Training Plan

## AutoCompress — Data & Training Plan

---

## 1. Dataset Plan

### 1.1 Supported Datasets

| Dataset | Classes | Images | Resolution | Use Case |
|---|---|---|---|---|
| **ImageNet-1K** (default) | 1,000 | 1.28M train / 50K val | 224×224 | Gold standard; pretrained weights directly compatible |
| **CIFAR-100** | 100 | 50K train / 10K val | 32×32 (→224 resize) | Faster iteration; more classes than CIFAR-10 |
| **CIFAR-10** | 10 | 50K train / 10K val | 32×32 (→224 resize) | Development / smoke tests; quickest benchmarking |
| **Custom** | User-defined | User-supplied | Any (resized to 224×224) | Production use case validation |

### 1.2 Dataset Sourcing

```bash
# ImageNet-1K (requires free academic registration at image-net.org)
# Download ILSVRC2012_devkit_t12.tar.gz + train/val tar files

# CIFAR-10 / CIFAR-100 (automatic via torchvision)
torchvision.datasets.CIFAR10(root='./data', download=True)
torchvision.datasets.CIFAR100(root='./data', download=True)

# Custom dataset: expects ImageFolder-compatible structure
/custom_data/
  train/
    class_A/  img1.jpg  img2.jpg ...
    class_B/  ...
  val/
    class_A/  ...
    class_B/  ...
```

### 1.3 Preprocessing Pipeline

```python
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

### 1.4 Calibration Set

- **Size:** 512 images, stratified across all classes
- **Purpose:** Used for PTQ calibration (collecting activation statistics) and SVD rank selection
- **Sampling:** Randomly sampled from the validation split at run-time (not stored separately)

---

## 2. Baseline Models

| Model | Parameters | Top-1 (ImageNet) | Size | Source |
|---|---|---|---|---|
| **ResNet-50** | 25.6M | 76.1% | ~98 MB | `torchvision.models.resnet50(pretrained=True)` |
| **MobileNetV2** | 3.4M | 71.8% | ~14 MB | `torchvision.models.mobilenet_v2(pretrained=True)` |
| **VGG-16** | 138M | 71.6% | ~528 MB | `torchvision.models.vgg16(pretrained=True)` |

> [!NOTE]
> These three models cover a range of architectures (residual, depthwise-separable, plain sequential) and sizes, ensuring the benchmark results generalize rather than being architecture-specific.

---

## 3. Per-Technique Training Configuration

### 3.1 Quantization — `quantize.py`

**Post-Training Quantization (PTQ) — default**

| Parameter | Value |
|---|---|
| Quantization scheme | `torch.quantization.quantize_dynamic` |
| Target layers | `nn.Linear`, `nn.Conv2d` |
| Data type | INT8 |
| Calibration | 512-image calibration set |
| Fine-tuning | **None** |

**Quantization-Aware Training (QAT) — optional `--qat` flag**

| Parameter | Value |
|---|---|
| Quantization backend | `fbgemm` (x86 CPU) or `qnnpack` (ARM/mobile) |
| Epochs | 10 |
| LR | 1e-5 (fine-tuning LR) |
| Optimizer | SGD + momentum 0.9 |
| Scheduler | Cosine annealing |

---

### 3.2 Pruning — `prune.py`

| Parameter | Value |
|---|---|
| Library | `torch-pruning` (structured, `MagnitudePruner`) |
| Importance metric | L1-norm of filter weights |
| Pruning ratio | 30% of filters per Conv2d layer (configurable) |
| Global vs layer-wise | Layer-wise by default; global with `--global-prune` |
| **Recovery fine-tuning** | **5 epochs** |
| LR | 1e-4 |
| Optimizer | Adam |
| Scheduler | CosineAnnealingLR |
| Batch size | 128 |
| Loss | Cross-entropy |

**Iterative pruning schedule:**
```
Epoch 0: Prune 30% of filters → measure accuracy drop
Epochs 1–5: Fine-tune recovery
Epoch 5: Re-measure → log recovery accuracy
```

---

### 3.3 Knowledge Distillation — `distill.py`

| Parameter | Value |
|---|---|
| Teacher | Original baseline model (frozen, eval mode) |
| Student | MobileNetV2 (default); configurable via `--student` |
| **Loss function** | `α·CE(student_logits, labels) + (1-α)·KL(student_soft/T ∥ teacher_soft/T)` |
| α (hard loss weight) | 0.3 |
| Temperature (T) | 4.0 |
| Epochs | Up to 90 (full training) |
| Early stopping | Patience = 5 epochs on val accuracy |
| LR | 0.01 (cosine decay to 1e-5) |
| Optimizer | SGD, momentum=0.9, weight_decay=1e-4 |
| Batch size | 256 |

**Loss derivation:**
```
L_total = α · CrossEntropy(logits_s, y_hard)
        + (1-α) · T² · KLDiv(
              softmax(logits_s / T),
              softmax(logits_t / T)
          )
```
The T² factor compensates for the softened gradient magnitudes.

---

### 3.4 Low-Rank Factorization — `lowrank.py`

| Parameter | Value |
|---|---|
| Method | Truncated SVD (scipy `svds`) |
| Target layers | All `nn.Conv2d` (reshaped to 2D) and `nn.Linear` |
| Rank ratio | 0.5 (keep top 50% singular values) |
| Skip layers | First Conv + last FC (critical layers) |
| Fine-tuning | **None** |
| Replacement | Original layer → two factored layers (U·Σ and V) |

---

## 4. Evaluation Protocol

### 4.1 Accuracy Evaluation

```python
def evaluate_accuracy(model, val_loader, device):
    model.eval()
    top1_correct, top5_correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # Top-1
            _, predicted = outputs.max(1)
            top1_correct += predicted.eq(labels).sum().item()
            # Top-5
            _, top5_pred = outputs.topk(5, dim=1)
            top5_correct += top5_pred.eq(labels.unsqueeze(1)).any(1).sum().item()
            total += labels.size(0)
    return top1_correct/total, top5_correct/total
```

### 4.2 Latency Evaluation

```python
def evaluate_latency(model, input_shape=(1,3,224,224), n_runs=100, device='cpu'):
    dummy = torch.randn(input_shape).to(device)
    model.eval()
    # Warm-up
    for _ in range(10):
        _ = model(dummy)
    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model(dummy)
        times.append(time.perf_counter() - start)
    return np.mean(times) * 1000, np.std(times) * 1000  # ms
```

### 4.3 Metrics Summary Table (per candidate)

| Metric | Formula | Logged In |
|---|---|---|
| Top-1 accuracy retention | `compressed_top1 / baseline_top1` | `candidate_results.csv` |
| Top-5 accuracy retention | `compressed_top5 / baseline_top5` | `candidate_results.csv` |
| Compression ratio | `baseline_size_mb / compressed_size_mb` | `candidate_results.csv` |
| Parameter reduction | `1 - compressed_params / baseline_params` | `candidate_results.csv` |
| Latency multiplier | `compressed_latency_ms / baseline_latency_ms` | `candidate_results.csv` |
| Export format latency | One row per format (TS / ONNX / TFLite) | `candidate_results.csv` |

---

## 5. Compute Requirements

| Task | Estimated Time | Hardware |
|---|---|---|
| Baseline evaluation (ImageNet) | ~15 min | CPU (i7) |
| PTQ / Low-Rank (per candidate) | ~20 min (inc. eval) | CPU |
| Pruning + 5-epoch fine-tuning | ~2–4 hrs | GPU (6GB+) |
| Distillation (90 epochs, early stop) | ~8–12 hrs | GPU (6GB+) |
| Full run (all 14 candidates, ImageNet) | ~2–3 days | GPU (RTX 3060+) |
| Full run (CIFAR-10, all candidates) | ~3–5 hrs | GPU (6GB+) |

> [!TIP]
> For development and smoke testing, use `--dataset cifar10 --search-depth 1` to get results in under an hour on a laptop GPU.

> [!IMPORTANT]
> Google Colab (free tier, T4 GPU) is sufficient for CIFAR-100 full runs and ImageNet partial runs. Use `--dataset imagenet-subset` with a 10-class mini-ImageNet for Colab-scale experiments.
