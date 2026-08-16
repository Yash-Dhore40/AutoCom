# 01 — Project Overview & Goals

## AutoCompress
### *An Adaptive, Multi-Technique Framework for Compressing Convolutional Neural Network Models*

| Field | Detail |
|---|---|
| **Project name** | AutoCompress |
| **Team** | Yash Dhore (RA2411026020078), Pushkar Sharma (RA2411026020105) |
| **Department** | CSE-AI/ML B, SRM University |
| **Review stage** | Zeroth Review — 03 Aug 2026 |
| **Scope** | Research + production-grade Python library + Jupyter benchmark reports |

---

## 1. Background & Motivation

Modern CNNs deliver strong accuracy on vision tasks, but their memory footprint, compute requirements, and energy consumption make direct deployment on resource-constrained devices impractical. A ResNet-50 checkpoint can be reduced **4–10×** in size with the right compression technique — yet most teams choose their technique by habit, not by evidence.

The proliferation of mobile AI (Android, iOS) and edge inference has made this problem urgent:

| Deployment target | Core constraint |
|---|---|
| Mobile apps (Android / iOS) | On-device inference; TFLite / CoreML size limits |
| Embedded / IoT | Tight RAM and flash budgets |
| Drones & robotics | Real-time perception under power limits |
| Edge cameras | Continuous inference on low-cost hardware |

---

## 2. Problem Statement

CNN compression today is **manual and inconsistent**:

1. **No single best technique** — Quantization, pruning, distillation, and low-rank factorization each win on different architectures.
2. **Trial-and-error workflow** — Engineers apply one method by hand, re-test, and repeat; the full option space is rarely explored.
3. **Inconsistent evaluation** — Accuracy, size, and latency are benchmarked differently across experiments, making cross-technique comparisons unreliable.

**Net effect:** Teams either under-compress (missing efficiency gains) or over-compress (losing accuracy they cannot recover), because no automated process tests the full technique space against a clear, shared priority ranking.

---

## 3. Proposed Solution

AutoCompress is an **automated, multi-technique compression pipeline** that:

1. Accepts any CNN checkpoint + validation dataset as input
2. Runs all valid technique combinations (single, pairs, triples) through a shared benchmark harness
3. Scores every candidate on accuracy retention, compression ratio, and inference latency — across three export formats
4. Selects and exports the best model, plus a full Jupyter comparison report

---

## 4. Goals & Success Metrics

### Primary Goal
Ship a reusable Python CLI/library that compresses any CNN checkpoint and delivers the best result — automatically, fairly, and reproducibly.

### Success Metrics

| Metric | Definition | Target |
|---|---|---|
| **Accuracy retention** | Compressed top-1 / baseline top-1 | ≥ 98% (default, user-configurable) |
| **Compression ratio** | Baseline size / compressed size | ≥ 2× minimum; ≥ 4× stretch goal |
| **Latency overhead** | Compressed time / baseline time | ≤ 1.5× on benchmark device |
| **Export coverage** | Formats validated | TorchScript + ONNX + TFLite all passing |
| **Benchmark fairness** | Same harness for all candidates | 100% — no technique gets special treatment |
| **CLI usability** | One command to run full pipeline | `autocompress run --model resnet50 --dataset imagenet` |

### Secondary Deliverables
- Auto-generated Jupyter Notebook report after every run (comparison table + plots)
- Documented reference results for ResNet-50, MobileNetV2, VGG-16 on ImageNet-1K
- Published accuracy / size / latency numbers for all 14+ pipeline candidates

---

## 5. Scope Boundaries

### In scope
- Quantization (PTQ, QAT), structured pruning, knowledge distillation, low-rank factorization (SVD)
- Full combinatorial pipeline search (depth 1, 2, 3)
- Export to TorchScript, ONNX, TFLite
- Latency benchmarking on CPU + Android ADB
- Jupyter Notebook benchmark report generation
- Accuracy floor + compression constraint filtering
- CIFAR-10, CIFAR-100, ImageNet-1K, and user-supplied datasets

### Out of scope (this version)
- Cloud/server deployment or model serving APIs
- Non-CNN architectures (ViT, Transformers, LLMs)
- Hardware-specific NAS
- Real-time mobile app integration (benchmarking via ADB only)
- Fairness / class-imbalance guardrails

---

## 6. Key References

| Paper | Relevance |
|---|---|
| Han et al. (2016). *Deep Compression*. ICLR | Pruning + Quantization + Huffman — foundational pipeline |
| Hinton et al. (2015). *Distilling the Knowledge in a Neural Network*. NeurIPS | Knowledge distillation methodology |
| Jacob et al. (2018). *Quantization and Training of Neural Networks*. CVPR | QAT with integer-arithmetic-only inference |
| Li et al. (2017). *Pruning Filters for Efficient ConvNets*. ICLR | Structured filter pruning |
| Denton et al. (2014). *Exploiting Linear Structure*. NeurIPS | SVD-based low-rank factorization |
