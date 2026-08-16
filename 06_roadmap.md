# 06 — Milestones & Phased Roadmap

## AutoCompress — Milestone-Driven Development Roadmap

> **Timeline approach:** Milestone-driven (not calendar-week fixed). Each phase unlocks the next only when its validation gate is passed. Estimated durations are provided as guidance.

---

## Roadmap Overview

```mermaid
graph LR
    M0["🔬 M0\nFoundation\n~2 weeks"] --> M1
    M1["⚙️ M1\nTechnique\nPrototype\n~3 weeks"] --> M2
    M2["📊 M2\nBenchmark\nHarness\n~2 weeks"] --> M3
    M3["🔗 M3\nPipeline\nIntegration\n~2 weeks"] --> M4
    M4["📱 M4\nMobile Export\n+ Report\n~2 weeks"] --> M5
    M5["🎓 M5\nFinal Review\n+ Paper Draft\n~1 week"]

    style M0 fill:#4f46e5,color:#fff
    style M1 fill:#0ea5e9,color:#fff
    style M2 fill:#10b981,color:#fff
    style M3 fill:#f59e0b,color:#fff
    style M4 fill:#ef4444,color:#fff
    style M5 fill:#8b5cf6,color:#fff
```

---

## Phase M0 — Foundation *(Est. ~2 weeks)*

### Goal
Set up the full development environment, load baseline models, and produce the first honest benchmark numbers before any compression.

### Tasks

| # | Task | Owner | Deliverable |
|---|---|---|---|
| M0.1 | Set up Python environment (conda/venv + deps) | Both | `requirements.txt`, `environment.yml` |
| M0.2 | Configure Git repo + CI (pytest on push) | Pushkar | `.github/workflows/ci.yml` |
| M0.3 | Implement `data/loaders.py` — ImageNet, CIFAR-10/100, Custom | Yash | `loaders.py` + unit tests |
| M0.4 | Implement `utils/model_loader.py` — torchvision + custom `.pth` | Pushkar | `model_loader.py` |
| M0.5 | Implement `benchmark.py` — accuracy, size, latency | Yash | `benchmark.py` + tests |
| M0.6 | Run baseline benchmarks for ResNet-50, MobileNetV2, VGG-16 | Both | `baseline_stats_*.json` |

### Validation Gate ✅
- [ ] `autocompress baseline --model resnet50 --dataset cifar10` runs without errors
- [ ] Baseline top-1 accuracy matches published numbers within ±0.5%
- [ ] Latency measurement is stable (std < 5% of mean across 100 runs)

---

## Phase M1 — Technique Prototype *(Est. ~3 weeks)*

### Goal
Implement all four compression techniques independently, each passing accuracy and size tests on CIFAR-10.

### Tasks

| # | Task | Owner | Deliverable |
|---|---|---|---|
| M1.1 | Implement `quantize.py` (PTQ) | Yash | Quantized ResNet-50 on CIFAR-10 |
| M1.2 | Implement `prune.py` (structured L1 pruning + 5-epoch FT) | Pushkar | Pruned ResNet-50 on CIFAR-10 |
| M1.3 | Implement `distill.py` (teacher→student KD) | Yash | Distilled MobileNetV2 on CIFAR-10 |
| M1.4 | Implement `lowrank.py` (truncated SVD) | Pushkar | Factorized ResNet-50 on CIFAR-10 |
| M1.5 | Define `techniques/base.py` — `CompressionTechnique` Protocol | Both | `base.py` interface |
| M1.6 | Unit tests for all 4 technique modules | Both | `test_techniques.py` |

### Validation Gate ✅
- [ ] Each technique compresses ResNet-50 on CIFAR-10 with ≥90% accuracy retention
- [ ] Each technique achieves ≥2× compression ratio
- [ ] All 4 tests pass in `pytest tests/test_techniques.py`
- [ ] All 4 modules implement `compress()` and `describe()` from the base Protocol

---

## Phase M2 — Benchmark Harness *(Est. ~2 weeks)*

### Goal
Validate that the benchmark engine produces fair, reproducible, consistent numbers across all four techniques.

### Tasks

| # | Task | Owner | Deliverable |
|---|---|---|---|
| M2.1 | Connect all 4 technique outputs to benchmark harness | Both | End-to-end per-technique test |
| M2.2 | Implement `candidate_results.csv` logging | Yash | CSV schema + writer |
| M2.3 | Validate benchmark consistency (run same candidate 3×, check variance) | Pushkar | Variance report |
| M2.4 | Implement `export.py` — TorchScript + ONNX export | Yash | `export.py` |
| M2.5 | Implement `utils/compat.py` — op compatibility checker | Pushkar | `compat.py` + known-bad-ops list |
| M2.6 | Run single-technique benchmark on all 3 baseline models × CIFAR-10 | Both | `candidate_results_M2.csv` |

### Validation Gate ✅
- [ ] All 4 × 3 = 12 single-technique benchmark rows populated in CSV
- [ ] Variance across 3 identical runs ≤ 2% for accuracy, ≤ 8% for latency
- [ ] TorchScript and ONNX exports load and run inference without errors
- [ ] Compat checker correctly flags unsupported ops (tested with a known-bad synthetic model)

---

## Phase M3 — Pipeline Integration *(Est. ~2 weeks)*

### Goal
Build the full Search Orchestrator + Selection Engine, enable stacked technique pipelines, and connect all modules end-to-end.

### Tasks

| # | Task | Owner | Deliverable |
|---|---|---|---|
| M3.1 | Implement `search.py` — candidate generator (depth 1, 2, 3) | Yash | 14-candidate list with ordering constraints |
| M3.2 | Implement run scheduler (sequential by default) | Pushkar | `scheduler.py` |
| M3.3 | Implement `select.py` — constraint filter + ranker | Yash | `select.py` |
| M3.4 | Implement `cli.py` — full Click CLI with all flags | Pushkar | `cli.py` |
| M3.5 | End-to-end test: `autocompress run` on CIFAR-10, all 14 candidates | Both | Passing test, CSV with 14 rows |
| M3.6 | Add `--search-depth 1` smoke test to CI | Pushkar | CI updated |

### Validation Gate ✅
- [ ] `autocompress run --model resnet50 --dataset cifar10 --search-depth 3` completes successfully
- [ ] Output directory contains all candidate model files
- [ ] Winner is selected and logged in `best/model_card.json`
- [ ] Accuracy ranking is verified correct (manually cross-checked)

---

## Phase M4 — Mobile Export & Report *(Est. ~2 weeks)*

### Goal
Add TFLite export, Android ADB benchmarking, and auto-generated Jupyter Notebook report. Run the full benchmark on ImageNet.

### Tasks

| # | Task | Owner | Deliverable |
|---|---|---|---|
| M4.1 | Implement TFLite export in `export.py` (ONNX → tf → tflite) | Yash | `.tflite` artifacts |
| M4.2 | Implement INT8 calibration for TFLite quantization | Yash | Calibration generator |
| M4.3 | Set up ADB benchmarking pipeline (script + result parser) | Pushkar | `adb_benchmark.sh` + Python parser |
| M4.4 | Implement `report.py` — Jupyter notebook generator | Pushkar | `benchmark_report.ipynb` template |
| M4.5 | Run full benchmark on ImageNet-1K (all 14 candidates, ResNet-50) | Both | Final `candidate_results_imagenet.csv` |
| M4.6 | Validate TFLite models on Android device | Both | ADB latency numbers in report |

### Validation Gate ✅
- [ ] TFLite model loads and runs on Android device without crash
- [ ] ADB latency numbers are recorded for the best candidate
- [ ] Jupyter notebook auto-generates correctly with real data (not mock)
- [ ] Plots render correctly and comparison table is populated
- [ ] At least one candidate achieves ≥98% accuracy retention AND ≥2× compression on ImageNet

---

## Phase M5 — Final Review & Paper Draft *(Est. ~1 week)*

### Goal
Polish all deliverables, finalize the benchmark report, and prepare the first draft of the research paper.

### Tasks

| # | Task | Owner | Deliverable |
|---|---|---|---|
| M5.1 | Run CIFAR-10 + CIFAR-100 + ImageNet benchmarks for all 3 baseline models | Both | Full reference results |
| M5.2 | Write README.md (installation, quick-start, full CLI reference) | Yash | `README.md` |
| M5.3 | Package CLI as installable (`pip install autocompress`) | Pushkar | `pyproject.toml`, `setup.cfg` |
| M5.4 | Draft research paper (methodology + results sections) | Both | `paper_draft.pdf` |
| M5.5 | Prepare final review presentation | Both | Presentation slides |
| M5.6 | Code freeze + tag v1.0.0 release on GitHub | Pushkar | `v1.0.0` release |

### Validation Gate ✅
- [ ] `pip install autocompress` works in a fresh virtual environment
- [ ] README quick-start successfully runs in under 5 minutes on CIFAR-10
- [ ] Reference results documented for ResNet-50 + MobileNetV2 + VGG-16 on all 3 datasets
- [ ] Paper draft: abstract, introduction, methodology, results, conclusion sections complete
- [ ] Final review presentation prepared

---

## Summary Milestone Table

| Milestone | Est. Duration | Key Output | Gate |
|---|---|---|---|
| M0 Foundation | ~2 weeks | Baseline benchmarks, dev env | Baseline accuracy matches published ±0.5% |
| M1 Technique Prototype | ~3 weeks | All 4 compression modules | 4/4 techniques pass accuracy + size tests |
| M2 Benchmark Harness | ~2 weeks | Fair comparison engine, ONNX export | Variance ≤2% accuracy, ≤8% latency |
| M3 Pipeline Integration | ~2 weeks | Full CLI, 14-candidate run | End-to-end `autocompress run` passes |
| M4 Mobile Export + Report | ~2 weeks | TFLite + Android ADB + Jupyter report | TFLite runs on Android, notebook auto-generates |
| M5 Final Review | ~1 week | Paper draft, pip package, v1.0.0 | pip install works, paper drafted |
| **Total** | **~12 weeks** | | |
