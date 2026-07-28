# ProTeGi vs SkillOpt 实验对比报告

## 实验设置

- **方法**: ProTeGi (Automatic Prompt Optimization with "Gradient Descent" and Beam Search, EMNLP 2023)
- **优化轮数**: 6 轮
- **Beam size**: 2-3
- **Optimizer**: gpt-5.5 (生成梯度和候选 prompt)
- **评测指标**: 与 SkillOpt 完全对齐 (Hard Metric)

## 结果

### gpt-5.5 (Predictor + Optimizer)

| 任务 | 指标 | SkillOpt | ProTeGi Baseline | ProTeGi R6 | ProTeGi 提升 |
|------|------|:---:|:---:|:---:|:---:|
| **SearchQA** | EM | **87.3%** | 78.6% | 86.4% | +7.8% |
| **SpreadsheetBench** | Acc | **80.7%** | 33.2% | 46.4% | +13.2% |
| **OfficeQA** | EM | **72.1%** | 46.5% | 39.5% | -7.0% |
| **DocVQA** | Hard Acc | 91.2% | 89.6% | **92.0%** | +2.4% |
| **LiveMath** | EM | 66.9% | 50.4% | **72.8%** | +22.4% |

### gpt-5.4-nano (Predictor) + gpt-5.5 (Optimizer)

| 任务 | 指标 | SkillOpt | ProTeGi Baseline | ProTeGi R6 | ProTeGi 提升 |
|------|------|:---:|:---:|:---:|:---:|
| **SearchQA** | EM | **74.8%** | 58.1% | 72.0% | +13.9% |
| **SpreadsheetBench** | Acc | **42.5%** | 26.4% | 33.6% | +7.2% |
| **OfficeQA** | EM | **50.0%** | 14.5% | 8.1% | -6.4% |
| **DocVQA** | Hard Acc | **80.2%** | 52.4% | 67.4% | +15.0% |
| **LiveMath** | EM | 27.2% | **31.2%** | 28.8% | -2.4% |

## 胜负统计

| | ProTeGi R6 赢 | SkillOpt 赢 |
|---|:---:|:---:|
| **gpt-5.5** | 2 (DocVQA, LiveMath) | 3 (SearchQA, Spreadsheet, OfficeQA) |
| **nano** | 0 | 5 |

## 评测对齐说明

| 任务 | 指标 | Test Set | Prompt 对齐 |
|------|------|:---:|:---:|
| SearchQA | EM (SQuAD normalize) | 1400 条 | SkillOpt system prompt |
| DocVQA | Hard Acc (ANLS ≥ 0.999) | 374 条 | SkillOpt system prompt + detail=auto |
| LiveMath | EM (parse_choice_label) | 125 条 | 无 theorem, multi-turn, `<answer>` 标签 |
| SpreadsheetBench | Acc (compare_workbooks) | 280 条 | SkillOpt 初始 skill |
| OfficeQA | EM (OfficeQA normalize) | 172 条 | oracle context + tools + max_turns=24 |

## ALFWorld 结果 (进行中)

### 实验配置

| | ProTeGi (本次) | SkillOpt |
|---|---|---|
| **优化方式** | Textual gradient + rewrite + beam search | Minibatch reflect + patch + selection gate |
| **Scoring** | Train set (39 games) | Selection set (valid_seen 140) + gate |
| **Rounds** | 6 | 4 epochs |
| **Beam size** | 4 | 1 (单 skill 迭代 patch) |
| **评估方式** | SkillOpt rollout (multiprocessing, 无 Ray) | 相同 |
| **初始 prompt** | `skillopt/envs/alfworld/skills/initial.md` | 相同 |

### Test Set (valid_unseen, 134 games)

| Method | Optimizer | Predictor | R1 | R2 | Best |
|--------|-----------|-----------|------|------|------|
| **SkillOpt** | gpt-5.5 | gpt-5.5 | — | — | **95.5%** |
| **SkillOpt** | gpt-5.5 | gpt-5.4-nano | — | — | **69.4%** |
| **ProTeGi** | gpt-5.5 | gpt-5.5 | 92.5% | 87.3% | 92.5% (R1) |
| **ProTeGi** | gpt-5.5 | gpt-5.4-nano | 44.0% | (running) | 44.0% (R1) |

### Train Set (39 games)

| Method | Predictor | R1 | R2 |
|--------|-----------|------|------|
| **ProTeGi** | gpt-5.5 | 87.2% | 87.2% |
| **ProTeGi** | gpt-5.4-nano | 53.8% | (running) |

### ALFWorld 分析

1. **ProTeGi + gpt-5.5 逐轮退化**：Test 92.5% → 87.3%，Train 不变（87.2%）。在小 train set 上过拟合，选出的 prompt 不泛化。
2. **ProTeGi + nano 远落后于 SkillOpt**：ProTeGi 44.0% vs SkillOpt 69.4%（差 25%）。
3. **SkillOpt 优势**：Selection gate 防止过拟合；trajectory 级反馈比 ProTeGi 的摘要级 error string 更丰富；patch 编辑保留好策略。

### 状态

- ProTeGi + gpt-5.5: Round 3 进行中 (~20h elapsed)
- ProTeGi + nano: Round 2 scoring 进行中 (~20h elapsed)
- 结果将在完成后更新。

---

## 关键发现

1. **SkillOpt 在大多数任务上显著优于 ProTeGi**，尤其是 SpreadsheetBench (80.7% vs 46.4%) 和 OfficeQA (72.1% vs 39.5%)
2. **ProTeGi 在 OfficeQA 上优化后退化**：两个模型都出现了负向优化，说明纯 prompt 优化在需要精确数值计算的任务上不仅无效还可能有害
3. **ProTeGi 在 DocVQA 和 LiveMath (5.5) 上略胜 SkillOpt**：可能因为这两个任务对 prompt 格式敏感，ProTeGi 的优化方向恰好有效
4. **SearchQA nano (72.0%) 接近 SkillOpt (74.8%)**：ProTeGi 在简单 QA 任务上接近 SkillOpt 水平
5. **强模型受益更多**：5.5 的 ProTeGi 提升普遍优于 nano，因为强模型能更好地遵循优化后的指令
6. **过拟合问题**：多个任务在中间轮次达到峰值后 R6 下降（如 SearchQA 5.5 最佳 R5=86.7% → R6=86.4%）
