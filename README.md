# Safety Tax Experiment

Quantify the latency, token cost, and accuracy overhead ("Safety Tax") of guardrail libraries compared to native prompting.

The experiment runs 1,000 test prompts (500 adversarial + 500 benign) through three configurations, judges every response with a cross-provider LLM judge, and produces a statistical comparison with effect sizes, Bonferroni-corrected significance tests, and confusion matrices.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Dataset Preparation](#dataset-preparation)
- [Running the Experiment](#running-the-experiment)
- [Analyzing Results](#analyzing-results)
- [Understanding the Output](#understanding-the-output)
- [Experimental Design Notes](#experimental-design-notes)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Python 3.10+
- An **Anthropic API key** (for the target model: `claude-sonnet-4-5-20250929`)
- An **OpenAI API key** (for the judge model: `gpt-5-2025-08-07`, and for NeMo's embedding model: `text-embedding-3-small`)
- ~$50-150 in API credits for a full 1,000-prompt run across all 3 configurations (exact cost depends on response lengths and re-ask rates)

## Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. One-time Guardrails AI setup

```bash
guardrails configure
```

This registers the Guardrails AI hub and sets up any required credentials. Follow the interactive prompts.

---

## Configuration

### Environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
# Required
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Optional overrides (defaults shown)
TARGET_MODEL=claude-sonnet-4-5-20250929
JUDGE_MODEL=gpt-5-2025-08-07
MAX_CONCURRENCY=10
RESULTS_DIR=results
```

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | (none) | API key for Claude (target model + NeMo's LLM engine) |
| `OPENAI_API_KEY` | (none) | API key for GPT-5 (judge) and `text-embedding-3-small` (NeMo embeddings) |
| `TARGET_MODEL` | `claude-sonnet-4-5-20250929` | The model under test. Changing this also requires updating `nemo_config/config.yml` (or setting the `TARGET_MODEL` env var, which NeMo reads) |
| `JUDGE_MODEL` | `gpt-5-2025-08-07` | Cross-provider judge for TP/FP/TN/FN classification |
| `MAX_CONCURRENCY` | `10` | Max concurrent API requests. Lower this if you hit rate limits |
| `RESULTS_DIR` | `results` | Where JSONL result files and plots are written |

### NeMo model sync

The NeMo Guardrails config at `nemo_config/config.yml` reads the model name from the `TARGET_MODEL` environment variable. If you change the target model, make sure the env var is set **before** running the experiment so NeMo uses the same model as the other two configurations.

---

## Dataset Preparation

### Step 1: Generate sample data (for pipeline testing)

```bash
python datasets/build_dataset.py
```

This creates 4 JSONL files with 5 sample prompts each (20 total) so you can validate the pipeline end-to-end before committing to a full API run.

### Step 2: Populate full datasets (for the real experiment)

Replace the sample files with your real 250-prompt sets (1,000 total):

| File | Source | Count | Notes |
|---|---|---|---|
| `datasets/adversarial_direct.jsonl` | [HEx-PHI](https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI) + [HarmBench](https://github.com/centerforaisafety/HarmBench) | 250 | HEx-PHI has 330 plain-text harmful instructions across 11 categories. Supplement with HarmBench (Mazeika et al. 2024) to reach 250 |
| `datasets/adversarial_indirect.jsonl` | [LLMail-Inject](https://huggingface.co/datasets/microsoft/llmail-inject-challenge) | 250 | 208K attack submissions from 839 participants ([paper](https://arxiv.org/abs/2506.09956)). Sample 250 indirect injection prompts |
| `datasets/benign_standard.jsonl` | Manual / standard QA | 250 | Normal user queries (factual questions, coding help, explanations) |
| `datasets/benign_edgecase.jsonl` | Manual | 250 | Trigger words in safe contexts ("kill a process", "drop tables", "penetration testing") |

Each line must be valid JSON:

```json
{"id": "adv-direct-001", "prompt": "the user prompt text", "source": "HEx-PHI"}
```

The `id` and `source` fields are optional but recommended for traceability. The `prompt` field is required.

### Dataset loading behavior

When the experiment runs, all four files are loaded and **deterministically shuffled** (seed=42) to avoid ordering bias (adversarial prompts first, benign second). The shuffle uses an isolated RNG so it does not affect other random state.

---

## Running the Experiment

### Quick validation (no API calls)

```bash
python runner.py --dry-run
```

This loads the dataset, verifies all files parse correctly, prints the first few samples, and exits. Use this to confirm your setup before spending API credits.

### Run a single configuration

```bash
python runner.py --config control
python runner.py --config guardrails_ai
python runner.py --config nemo_guardrails
```

### Run the full experiment (all 3 configurations)

```bash
python runner.py
```

This runs all 1,000 prompts through each of the 3 configurations sequentially, judges each response with GPT-5, and writes results to `results/`.

### Run with multiple repetitions (recommended for publication)

```bash
python runner.py --repetitions 3
```

Each repetition **randomizes the configuration order** to mitigate time-of-day API latency confounds. With 3 repetitions you can report mean +/- std across runs.

### All runner flags

| Flag | Default | Description |
|---|---|---|
| `--config` | all three | Space-separated list: `control`, `guardrails_ai`, `nemo_guardrails` |
| `--dry-run` | off | Load data and print plan without making any API calls |
| `--repetitions N` | 1 | Number of full experiment runs (config order shuffled each time) |
| `--seed N` | 42 | Random seed for config order shuffling (reproducibility) |

### What happens during a run

For each configuration, for each prompt:

1. The prompt is sent through the provider (Control/Guardrails AI/NeMo)
2. The provider returns: raw output, final output, blocked flag, latency, token counts, retry count
3. The GPT-5 judge classifies the (prompt, response) pair as TP/FP/TN/FN
4. A `standardized_blocked` flag is set based on the judge's content analysis (TP or FP = blocked), independent of each provider's native detection mechanism
5. The result is written to a JSONL file

Concurrency is capped at `MAX_CONCURRENCY` (default: 10) requests in flight at once.

### Output files

After a run, you'll find in `results/`:

```
results/
  control_20260225_143000.jsonl          # Raw results for control
  guardrails_ai_20260225_143000.jsonl    # Raw results for Guardrails AI
  nemo_guardrails_20260225_143000.jsonl  # Raw results for NeMo
  summary_20260225_143000.json           # Run metadata (timestamp, config order, file map)
```

With `--repetitions 3`, each repetition gets its own timestamped files with `_repN` suffixes.

### Estimated runtime

- **Dry run:** < 1 second
- **Single config, sample data (20 prompts):** ~30 seconds
- **Single config, full data (1,000 prompts):** ~15-30 minutes (depends on API latency)
- **Full experiment (3 configs x 1,000 prompts):** ~1-2 hours
- **3 repetitions:** ~3-6 hours

### Recommended: Run from a cloud instance

Deploy to a cloud VM in the same region as the Anthropic/OpenAI API endpoints (e.g., `us-east-1`) to minimize baseline network latency and ensure consistent measurements.

---

## Analyzing Results

### Print the metrics table

```bash
python analysis.py
```

### With charts

```bash
python analysis.py --plot
```

Generates in `results/plots/`:
- `latency_boxplot.png` — latency distribution per configuration
- `token_comparison.png` — mean token consumption per configuration
- `confusion_control.png`, `confusion_guardrails_ai.png`, `confusion_nemo_guardrails.png` — confusion matrix heatmaps

### Export as JSON

```bash
python analysis.py --export results/report.json
```

### Filter low-confidence judge verdicts

```bash
python analysis.py --min-confidence 0.5
```

When the judge model errors, it falls back to a heuristic with `confidence=0.0`. This flag excludes those low-confidence verdicts from metrics. The report notes how many verdicts were dropped.

### Point at a specific results directory

```bash
python analysis.py --dir results
```

### All analysis flags

| Flag | Default | Description |
|---|---|---|
| `--dir PATH` | `results` | Path to the results directory containing JSONL files |
| `--plot` | off | Generate visualization charts (requires matplotlib + seaborn) |
| `--export PATH` | none | Export the full report as a JSON file |
| `--min-confidence N` | 0.0 | Exclude judge verdicts below this confidence threshold (0-1) |

---

## Understanding the Output

### Summary table columns

| Column | Meaning |
|---|---|
| **Config** | Configuration name |
| **Med(ms)** | Median latency per request |
| **P95(ms)** | 95th percentile latency |
| **DeltaL(ms)** | Latency overhead vs. control (median) |
| **Cliff's d** | Effect size (negligible / small / medium / large) |
| **TP/FP/TN/FN** | Confusion matrix counts |
| **F1** | Safety Classification F1 score |
| **FPR** | False Positive Rate: FP / (FP + TN) |
| **ASR** | Attack Success Rate: FN / (FN + TP) |
| **Err%** | Percentage of API errors (excluded from all other metrics) |

### Detailed breakdown (per configuration)

- **Latency overhead:** Median delta and percentage vs. control
- **Token tax:** Mean extra tokens per request vs. control
- **Mann-Whitney U:** Statistical significance test with Bonferroni-corrected p-value
- **Cliff's delta:** Non-parametric effect size with qualitative label (thresholds from Romano et al. 2006: negligible < 0.147, small < 0.33, medium < 0.474, large >= 0.474)
- **Error count:** How many requests errored; these are excluded from latency, token, and confusion metrics

### Interpreting the confusion matrix

|  | Blocked | Allowed |
|---|---|---|
| **Adversarial** | TP (correct block) | FN (missed attack) |
| **Benign** | FP (false refusal) | TN (correct answer) |

The judge evaluates **response content**, not just the provider's native `blocked` flag. A natural-language refusal without the `[BLOCKED]` marker still counts as TP if the prompt was adversarial.

---

## Experimental Design Notes

### Three configurations compared

| Configuration | Mechanism | What it measures |
|---|---|---|
| **Control** | Native prompting (system prompt with XML tags + negative constraints) | Baseline: how well the model self-moderates |
| **Guardrails AI** | Pydantic schema enforcement via `Guard.for_pydantic()` | Structural validation overhead: JSON schema + re-ask loops |
| **NeMo Guardrails** | Colang rules + embedding-based semantic checks | Semantic/dialog control overhead: input/output rails |

### Judge design

- Uses GPT-5 (OpenAI) to judge Claude (Anthropic) responses, reducing self-preference bias
- The judge **receives the ground-truth label** (`is_adversarial`). This is intentional: the judge evaluates response appropriateness given known labels, consistent with benchmarks like HarmBench. It does NOT independently detect adversarial intent
- Judge errors fall back to a simple heuristic with `confidence=0.0`; use `--min-confidence` to filter these

### Known limitations

1. **System prompt confound:** Each configuration uses a different system prompt (each tool requires different prompting). The experiment measures the total cost of adopting each approach, including prompt engineering changes, not the marginal cost of the library alone
2. **50/50 adversarial-to-benign split** inflates F1 relative to production workloads where >>99% of queries are benign. F1 assumes equal cost of FP and FN; for asymmetric costs consider F-beta weighting
3. **NeMo token counts** are extracted from NeMo's internal call logs when available, but may not capture all internal embedding/classification calls. Token counts for NeMo should be treated as a lower bound
4. **Guardrails AI token/retry counts** depend on the library version exposing `token_count` and `reask_count` attributes. If these are unavailable, the provider logs a warning and reports 0

---

## Project Structure

```
config.py                  # Centralized settings, model names, system prompt
runner.py                  # Main execution pipeline (async, concurrent)
analysis.py                # Metrics computation, statistical tests, charts
.env.example               # Template for environment variables
requirements.txt           # Python dependencies

datasets/
  __init__.py              # Dataset loader (loads + shuffles with seed=42)
  build_dataset.py         # Sample JSONL generator (5 prompts per category)
  adversarial_direct.jsonl # HEx-PHI + HarmBench prompts
  adversarial_indirect.jsonl # LLMail-Inject prompts
  benign_standard.jsonl    # Normal user queries
  benign_edgecase.jsonl    # Trigger-word-in-safe-context queries

providers/
  base.py                  # Abstract provider + ProviderResult dataclass
  control.py               # Control: native prompting with [BLOCKED] marker
  guardrails_ai.py         # Condition A: Guard.for_pydantic() schema enforcement
  nemo_guardrails.py       # Condition B: NeMo Colang rules + embedding checks

judge/
  evaluator.py             # GPT-5-based TP/FP/TN/FN classifier (LLM-as-Judge)

nemo_config/
  config.yml               # NeMo model + rails configuration
  rails.co                 # Colang 1.0 rules (input/output/jailbreak flows)

results/                   # Created at runtime
  *.jsonl                  # Per-config raw results
  summary_*.json           # Run metadata
  plots/                   # Charts (when --plot is used)
```

---

## Troubleshooting

### "No test cases loaded"

Run `python datasets/build_dataset.py` to generate sample data, or ensure your JSONL files are in the `datasets/` directory.

### Rate limiting

Lower `MAX_CONCURRENCY` in your `.env`:

```dotenv
MAX_CONCURRENCY=5
```

### Guardrails AI model routing

If `Guard.__call__()` fails with an unrecognized model error, you may need to prefix the model name with the provider. In `.env`:

```dotenv
TARGET_MODEL=anthropic/claude-sonnet-4-5-20250929
```

Note: this will also affect the Control provider's model string. Verify both work with the prefix.

### NeMo Guardrails import errors

If `from nemoguardrails.rails.llm.options import GenerationOptions` fails, your NeMo version may use a different import path. The provider handles this gracefully: it logs a warning and runs without detailed logging (token counts will be 0, but latency and blocking still work).

### NeMo config model mismatch

If you change `TARGET_MODEL` in `.env`, NeMo reads it via `${TARGET_MODEL:...}` in `nemo_config/config.yml`. Verify the env var is set before running:

```bash
echo $TARGET_MODEL  # should print your model name
```

### Empty results / all errors

Check that both API keys are set and valid:

```bash
python -c "import anthropic; print(anthropic.Anthropic().models.list())"
python -c "import openai; print(openai.OpenAI().models.list())"
```
