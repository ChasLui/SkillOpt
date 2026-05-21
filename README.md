# SkillOpt

**Executive Strategy for Self-Evolving Agent Skills**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

*Train agent skills like you train neural networks — with epochs, learning rates, and validation gates — but without touching model weights.*

---

## What is SkillOpt?

SkillOpt is a framework for optimizing a natural-language **skill document** through iterative rollout, reflection, editing, and gated validation.

It does **not** fine-tune model parameters. Instead, it treats the skill document as the optimization target:

- The **student** model executes tasks with the current skill
- The **teacher** model analyzes trajectories and proposes edits
- The framework merges, ranks, applies, and validates those edits
- Only validated skill updates are kept

| Deep Learning | SkillOpt |
|---|---|
| Model weights | Skill document (Markdown) |
| Forward pass | Rollout (student executes tasks) |
| Loss computation | Reflect (teacher analyzes trajectories) |
| Gradient | Edit patches (proposed skill improvements) |
| Gradient clipping | Edit ranking & selection (`learning_rate`) |
| Weight update | Patch application to skill document |
| Validation | Gated evaluation on held-out split |
| Learning rate schedule | `lr_scheduler`: cosine, linear decay |
| Epochs | Multi-epoch training with slow update & meta skill |

---

## Method Overview

### Optimization Target

Each run maintains a mutable markdown skill document. The framework repeatedly improves that document instead of changing model parameters.

This gives a training-style loop for prompt / policy optimization:

1. Roll out the current skill on a batch of tasks.
2. Reflect on failures and successes.
3. Merge patch proposals into a coherent candidate update.
4. Rank and select a bounded number of edits.
5. Apply those edits to produce a candidate skill.
6. Validate the candidate skill on a held-out selection split.
7. Keep the update only if the gate accepts it.

### Per-Step Pipeline

Every training step executes the following pipeline in `skillopt/engine/trainer.py`:

1. **Rollout**
   The student model runs a batch of tasks using the current skill.

2. **Reflect**
   The teacher analyzes minibatches of trajectories and emits raw patches.
   Failure-driven and success-driven patches are tracked separately.

3. **Aggregate**
   Raw patches are merged hierarchically. Metadata such as `support_count` and `source_type` is carried into the merged patch so later ranking can use it.

4. **Select**
   The teacher ranks the merged edit pool and keeps up to `edit_budget` edits.

5. **Update**
   The selected edits are applied to the skill document. The framework records an `edit_apply_report.json` so you can see which edits actually landed, which were skipped, and why.

6. **Evaluate / Gate**
   The candidate skill is evaluated on the selection split. A candidate update is accepted only if it improves over the current selection score; a new global best is tracked separately.

### Within-Epoch Memory

Inside an epoch, the trainer maintains a step buffer containing:

- Compact failure-pattern summaries from previous steps
- Rejected edits and their score deltas

That context is fed back into later reflection calls so the teacher can avoid repeating ineffective edits and can focus on unsolved error patterns.

### Epoch-Level Mechanisms

#### Slow Update

At the end of each epoch, `slow_update` compares the previous epoch's terminal skill and current epoch's terminal skill on a sampled train subset. It then writes longitudinal guidance into a protected slow-update region inside the skill document.

This guidance is **not** blindly written through — it is converted into a candidate skill and sent through the same selection gate as step-level updates.

#### Meta Skill

`meta_skill` is teacher-side cross-epoch memory. It does not directly edit the current skill. Instead, it writes a compact memory artifact describing longer-term patterns across adjacent epochs. That memory is loaded into later reflection / merge / ranking calls as extra context.

#### Meta Reflect

`meta_reflect` runs at epoch end over the step history of the current epoch. It looks at accepted and rejected directions from the whole epoch, proposes higher-level patch edits, applies them to a meta candidate, and then sends that candidate through the same selection gate.

---

## Quick Start

### Install

```bash
git clone https://github.com/AgenticOpt/SkillOpt.git
cd SkillOpt
pip install -e .
```

### Configure API Credentials

```bash
cp .env.example .env
# Edit .env with your API credentials, then:
source .env
```

**Azure OpenAI** (API key or managed identity):
```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-key"
# Or use managed identity: set azure_openai_auth_mode=managed_identity in config
```

**OpenAI** directly:
```bash
export OPENAI_API_KEY="sk-..."
```

**Anthropic Claude**:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Qwen (local vLLM)**:
```bash
export QWEN_CHAT_BASE_URL="http://localhost:8000/v1"
export QWEN_CHAT_MODEL="Qwen/Qwen3.5-4B"
```

### Run Training

```bash
python scripts/train.py --config configs/searchqa/default.yaml
```

---

## Configuration

SkillOpt uses a hierarchical YAML configuration system. Each benchmark config inherits from `configs/_base_/default.yaml`.

### Configuration Structure

```yaml
model:
  teacher_backend: openai_chat     # openai_chat | claude_chat | qwen_chat
  student_backend: openai_chat     # openai_chat | claude_chat | codex_exec | qwen_chat
  teacher: gpt-5.5                 # teacher model deployment name
  student: gpt-5.5                 # student model deployment name
  reasoning_effort: medium         # low | medium | high

train:
  num_epochs: 4
  batch_size: 40
  seed: 42

gradient:
  minibatch_size: 8                # trajectories per reflection call
  analyst_workers: 16              # parallel reflection workers
  use_deep_reflect: false          # deep multi-turn probing
  deep_reflect_failures: 4
  deep_reflect_successes: 2

optimizer:
  learning_rate: 4                 # max edits per step (edit_budget)
  min_learning_rate: 2             # min edits for decay schedulers
  lr_scheduler: cosine             # constant | linear | cosine | autonomous
  skill_update_mode: patch         # patch | rewrite_from_suggestions | full_rewrite_minibatch
  use_slow_update: true
  use_meta_skill: true
  use_meta_reflect: false

evaluation:
  use_gate: true                   # gated validation (always recommended)

env:
  name: ""                         # benchmark name
  skill_init: ""                   # path to initial skill document
  split_mode: ratio                # ratio | split_dir
  split_ratio: "2:1:7"            # train:val:test
```

### CLI Overrides

Override any config key from the command line:

```bash
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --cfg-options model.teacher_backend=openai_chat \
                model.student_backend=codex_exec \
                train.batch_size=40 \
                optimizer.learning_rate=4

# Legacy flat overrides also work for common keys:
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --backend azure_openai \
  --teacher_model gpt-5.5 \
  --student_model gpt-5.5 \
  --reasoning_effort medium
```

---

## Model Backends

All model access goes through the unified backend router in `skillopt/model/`.

| Backend | Use case | Config key |
|---|---|---|
| `openai_chat` | Azure OpenAI / OpenAI API | teacher / student |
| `claude_chat` | Anthropic Claude | teacher / student |
| `codex_exec` | Codex execution harness | student only |
| `qwen_chat` | Local Qwen via vLLM | teacher / student |

Separate teacher/student endpoints are supported:

```yaml
model:
  teacher_backend: openai_chat
  student_backend: codex_exec
  teacher: gpt-5.5
  student: gpt-5.5-codex
```

---

## Data Splits

SkillOpt supports two split modes:

**Ratio split** — auto-generate from raw data:
```bash
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --split_mode ratio \
  --data_path /path/to/searchqa_data.json
```

**Pre-split directory** — consume prepared splits:
```bash
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --split_mode split_dir \
  --split_dir /path/to/searchqa_split
```

---

## Supported Benchmarks

| Benchmark | Type | Config |
|---|---|---|
| SearchQA | QA | `configs/searchqa/default.yaml` |
| SpreadsheetBench | Code generation | `configs/spreadsheetbench/default.yaml` |
| ALFWorld | Embodied agent | `configs/alfworld/default.yaml` |
| DocVQA | Document QA | `configs/docvqa/default.yaml` |
| OfficeQA | Tool-augmented QA | `configs/officeqa/default.yaml` |
| SealQA | Tool-augmented QA | `configs/sealqa/default.yaml` |
| BabyVision | Vision QA | `configs/babyvision/default.yaml` |
| LiveMathematicianBench | Math | `configs/livemathematicianbench/default.yaml` |
| MathVerse | Multimodal math | `configs/mathverse/default.yaml` |
| MMRB | Multimodal reasoning | `configs/mmrb/default.yaml` |
| SWEBench | Software engineering | `configs/swebench/default.yaml` |

---

## Running Training

Basic training:

```bash
python scripts/train.py --config configs/searchqa/default.yaml
```

Exec harness (Codex student):

```bash
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --teacher_backend openai_chat \
  --student_backend codex_exec \
  --teacher_model gpt-5.5 \
  --student_model gpt-5.5-codex \
  --use_deep_reflect true \
  --skill_update_mode rewrite_from_suggestions
```

SWEBench:

```bash
python scripts/train.py \
  --config configs/swebench/default.yaml \
  --cfg-options env.dataset_name=lite env.split_ratio=2:1:7
```

### Eval Only

Evaluate a specific skill without training:

```bash
python scripts/eval_only.py \
  --config configs/searchqa/default.yaml \
  --skill skillopt/envs/searchqa/skills/initial.md
```

---

## Output Structure

Each run writes a structured output directory:

```
outputs/<run_name>/
├── config.json              # Flattened runtime config
├── history.json             # Per-step history records
├── runtime_state.json       # Resume state (for auto-resume)
├── best_skill.md            # Current best validated skill
├── skills/skill_vXXXX.md   # Skill snapshot per step
├── steps/step_XXXX/        # Per-step artifacts
│   ├── merged_patch.json
│   ├── ranked_edits.json
│   ├── candidate_skill.md
│   ├── edit_apply_report.json
│   ├── rewrite_result.json  # when rewrite mode is enabled
│   └── selection_eval/
├── slow_update/epoch_XX/
├── meta_skill/epoch_XX/
└── meta_reflect/epoch_XX/
```

### Resume Behavior

The trainer resumes from `runtime_state.json` when present. That state tracks:

- Last completed step
- Current skill path and score
- Best skill path and score
- Origin tags for current and best skill

---

## Extending SkillOpt

### Add a New Benchmark

1. Create `skillopt/envs/<your_env>/` with:
   - `adapter.py` — implements `EnvAdapter`
   - `dataloader.py` — data loading logic
   - `rollout.py` — student execution logic
   - `skills/initial.md` — initial skill document
2. Add a config at `configs/<your_env>/default.yaml`
3. Register in `skillopt/envs/__init__.py`

See `skillopt/envs/_template/` for a scaffold.

### Add a New Model Backend

Implement a backend in `skillopt/model/` following the interface in `skillopt/model/common.py`, then register it in `skillopt/model/router.py`.

---

## WebUI

Launch the monitoring dashboard (optional):

```bash
pip install -e ".[webui]"
python -m skillopt_webui.app
```

Provides browser-based config selection, training launch, and real-time log monitoring.

---

## Minimal Setup

```bash
conda create -n skillopt python=3.11
conda activate skillopt
pip install -e .
```

Depending on the benchmark, you may also need:

```bash
pip install datasets gymnasium numpy
```

For SWEBench, you also need a working Docker environment plus the SWE-bench harness dependencies.

---

## Citation

```bibtex
@article{skillopt2026,
  title={SkillOpt: Executive Strategy for Self-Evolving Agent Skills},
  author={SkillOpt Team},
  year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
