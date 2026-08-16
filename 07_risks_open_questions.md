# 07 — Risks & Open Questions

## AutoCompress — Risk Register & Open Decisions

---

## 1. Risk Register

### Risk Severity Matrix

| ID | Risk | Category | Likelihood | Severity | Priority |
|---|---|---|---|---|---|
| R1 | ONNX → TFLite conversion fails for certain operators | Technical | 🔴 High | 🔴 High | **Critical** |
| R2 | ImageNet dataset access / storage (~150GB) | Logistics | 🟡 Medium | 🟡 Medium | **High** |
| R3 | Full combinatorial search (14 candidates) is too slow on available hardware | Performance | 🔴 High | 🟡 Medium | **High** |
| R4 | Distillation student doesn't converge or underperforms teacher severely | ML | 🟡 Medium | 🟡 Medium | **High** |
| R5 | Low-rank factorization incompatible with TFLite (custom layer replacement) | Technical | 🟡 Medium | 🔴 High | **High** |
| R6 | Accuracy retention target (≥98%) too strict — no candidate passes | ML | 🟡 Medium | 🟡 Medium | **Medium** |
| R7 | ADB benchmarking requires physical Android device the team may not own | Logistics | 🟢 Low | 🟡 Medium | **Medium** |
| R8 | torch-pruning API changes between PyTorch versions | Technical | 🟢 Low | 🟡 Medium | **Medium** |
| R9 | QAT (quantization-aware training) adds significant complexity vs. PTQ | Scope | 🟡 Medium | 🟢 Low | **Low** |
| R10 | Paper draft scope too ambitious for one semester | Scope | 🟢 Low | 🟢 Low | **Low** |

---

## 2. Detailed Risk Analysis & Mitigations

### R1 — ONNX → TFLite conversion failures *(Critical)*

**What can go wrong:** PyTorch ops that export fine to ONNX (opset 17) may not have a corresponding TFLite implementation. Particularly risky for:
- Custom SVD-decomposed layers (lowrank.py)
- BatchNorm folded differently
- Dynamic shapes

**Impact:** TFLite export fails silently or produces incorrect results, breaking the mobile benchmark path.

**Mitigation:**
1. Run `compat.py` op-checker immediately after every ONNX export
2. For failed candidates, fall back to ONNX Runtime (CPU) latency and document this in the report
3. Freeze model to static shapes before export (`torch.jit.trace` instead of `script` for export)
4. Keep a pinned conversion environment: `onnx==1.15`, `onnx-tf==1.14`, `tensorflow==2.14`

---

### R2 — ImageNet storage / access *(High)*

**What can go wrong:** ImageNet-1K requires ~150GB and an academic registration. Download may be slow or blocked.

**Impact:** Full benchmark runs are blocked; delays all downstream milestones.

**Mitigation:**
1. **Immediate fallback:** CIFAR-100 is used for all development benchmarks (automatic download, 170MB)
2. **ImageNet substitute:** Use ImageNet-1K 10-class subset (Tiny ImageNet or manually curated 10-class split, ~1.2GB)
3. **Cloud option:** Kaggle provides ImageNet access in Kaggle Notebooks without download
4. Document which dataset was used for every result row in the CSV

---

### R3 — Combinatorial search is computationally expensive *(High)*

**What can go wrong:** 14 candidates × 3 export formats × distillation training (up to 90 epochs each) = potentially weeks of compute.

**Impact:** Full benchmark never completes; project stalls at M4.

**Mitigation:**
1. **Tiered benchmarking:**
   - Phase 1: Run all 14 candidates on CIFAR-10 (fast, ~3–5 hrs)
   - Phase 2: Run top-5 candidates on ImageNet (selected from CIFAR-10 results)
   - Phase 3: Full run on ImageNet only if time permits
2. **Early stopping for distillation:** Patience=5 epochs; prune search tree after 3 non-improving candidates
3. **Parallel execution:** Run non-dependent candidates in parallel using `multiprocessing.Pool`
4. **GPU access:** Apply for Google Cloud Education credits (~$300 free)

---

### R4 — Knowledge distillation convergence failure *(High)*

**What can go wrong:** The student architecture (MobileNetV2) may fail to match teacher accuracy, especially on ImageNet where the capacity gap is large.

**Impact:** Distillation-containing candidates always fail the accuracy floor; reduces search space to 10 candidates.

**Mitigation:**
1. Use a less aggressive student: start with ResNet-18 (not MobileNetV2) for distillation from ResNet-50 teacher
2. Tune temperature T ∈ {2, 4, 8} and α ∈ {0.1, 0.3, 0.5} in a quick hyperparameter sweep before full run
3. If distillation consistently fails the floor, reduce the floor for distillation candidates to 95% and document this as a technique-specific override
4. Refer to Hinton et al. (2015) hyperparameters as starting point

---

### R5 — Low-rank layers incompatible with TFLite *(High)*

**What can go wrong:** `lowrank.py` replaces standard Conv2d with two smaller Conv2d layers (SVD factors). While standard, the resulting model graph may not export cleanly through the ONNX → TF → TFLite chain.

**Impact:** All candidates containing LowRank have no TFLite export; only ONNX + TorchScript available.

**Mitigation:**
1. Test LowRank export at M2 (Phase 2) — earliest possible detection
2. If TFLite fails, serve ONNX export for LowRank candidates and document the limitation
3. Alternative: After SVD factorization, re-merge weights back into a single Conv2d before export (weights-level, not graph-level — loses the size benefit but makes export safe)

---

### R6 — Accuracy floor too strict *(Medium)*

**What can go wrong:** With default 98% accuracy floor, aggressive techniques (heavy pruning, deep stacks) are eliminated. Only quantization-only candidates pass. The project ends up benchmarking one technique effectively.

**Impact:** The "comparative" value of the project is reduced.

**Mitigation:**
1. Always report the full candidate table regardless of floor — floor only affects the *selected winner*, not the *report*
2. Expose `--accuracy-floor` as a CLI arg with 98% as default — users can relax to 95% or 90%
3. Frame this in the paper as: "We show which techniques survive a strict 98% retention gate, and which require relaxing it"

---

### R7 — No physical Android device available *(Medium)*

**What can go wrong:** ADB benchmarking requires a connected Android device with developer mode enabled.

**Impact:** Android latency numbers are unavailable; TFLite deployment story is theoretical.

**Mitigation:**
1. Use Android emulator for functional validation (latency numbers not representative — documented)
2. Use ONNX Runtime CPU latency as the published benchmark; mark Android ADB as "stretch goal"
3. Request loan of a device from college lab or borrow a team member's personal phone
4. Use [TFLite Benchmark Tool web interface](https://ai.google.dev/edge/litert/inference) for proxy measurements

---

## 3. Open Design Questions

These are decisions that are **not yet resolved** and will affect implementation:

| # | Question | Options | Recommended |
|---|---|---|---|
| **OQ1** | Should the search use a fixed hyperparameter per technique or sweep? | Fixed defaults vs. grid search | Fixed defaults for MVP; add sweep mode as flag |
| **OQ2** | Should the Jupyter report be auto-executed (`nbconvert`) or left as-run? | Auto-execute vs. manual | Auto-execute via `nbconvert --to html` for sharing |
| **OQ3** | What student architecture to use for distillation if teacher is VGG-16? | ResNet-18 / MobileNetV2 / EfficientNet-B0 | ResNet-18 (most compatible architecture) |
| **OQ4** | Should LowRank apply globally to all layers or skip the first/last? | Global vs. selective | Skip first+last layer (empirically more stable) |
| **OQ5** | Should the CLI support a `--resume` flag to skip already-run candidates? | Yes / No | Yes — essential for long-running searches |

---

## 4. Technical Tradeoffs

| Tradeoff | Option A | Option B | Decision |
|---|---|---|---|
| Speed vs. coverage | Run subset of candidates fast | Run all 14 candidates fully | **Tiered** (fast sweep on CIFAR, full on ImageNet top-N) |
| Export simplicity vs. reach | ONNX only | All 3 formats | **All 3** (per user decision) — accept conversion risk |
| Fine-tuning consistency vs. fairness | Same FT budget for all | Technique-natural FT | **Technique-natural** (per user decision) — best results per technique |
| Accuracy floor flexibility | Hard-coded 98% | User-configurable | **User-configurable** with 98% default (per user decision) |
| Search depth vs. compute | Depth 1 only | Full depth 3 | **Full depth 3** (per user decision) — tiered execution to manage cost |

---

## 5. Dependency Versions (Pinned for Stability)

```txt
# requirements.txt
torch==2.2.0
torchvision==0.17.0
torch-pruning==1.3.5
onnx==1.15.0
onnxruntime==1.17.0
onnx-tf==1.14.0
tensorflow==2.14.0
numpy==1.26.4
pandas==2.2.0
matplotlib==3.8.3
click==8.1.7
nbformat==5.9.2
pytest==8.0.2
scipy==1.12.0
```

> [!CAUTION]
> The `onnx-tf` package is known to have compatibility issues with TensorFlow versions above 2.14. Do **not** upgrade TensorFlow past 2.14 without testing the full ONNX → TFLite chain.
