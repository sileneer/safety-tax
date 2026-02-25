"""
runner.py — Main execution pipeline for the Safety Tax experiment.

Orchestrates:
  1. Dataset loading
  2. Running all 1000 prompts through each of the 3 configurations
  3. Judging each result with gpt-5-2025-08-07
  4. Saving raw results to JSONL for analysis

Usage:
    python runner.py                      # Run all configurations
    python runner.py --config control     # Run only the control
    python runner.py --dry-run            # Load data, skip API calls
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from datetime import datetime
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio

import config
from datasets import load_all_datasets, TestCase
from providers import ControlProvider, GuardrailsAIProvider, NeMoGuardrailsProvider
from providers.base import BaseProvider, ProviderResult
from judge import JudgeEvaluator


# ── Result schema ─────────────────────────────────────────────────────────────
def result_to_dict(
    case: TestCase,
    config_name: str,
    result: ProviderResult,
    verdict: dict | None,
) -> dict:
    return {
        "test_id": case.id,
        "category": case.category,
        "is_adversarial": case.is_adversarial,
        "prompt": case.prompt,
        "config": config_name,
        "raw_output": result.raw_output,
        "final_output": result.final_output,
        "blocked": result.blocked,
        "standardized_blocked": result.standardized_blocked,
        "latency_ms": result.latency_ms,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "total_tokens": result.total_tokens,
        "retries": result.retries,
        "error": result.error,
        "verdict": verdict,
    }


# ── Concurrency-limited executor ──────────────────────────────────────────────
async def run_single(
    semaphore: asyncio.Semaphore,
    provider: BaseProvider,
    case: TestCase,
    judge: JudgeEvaluator,
) -> dict:
    async with semaphore:
        result = await provider.process(case.prompt)

        # Judge the result
        verdict_obj = await judge.evaluate(
            prompt=case.prompt,
            is_adversarial=case.is_adversarial,
            response=result.final_output,
            blocked=result.blocked,
        )
        verdict = {
            "classification": verdict_obj.classification,
            "reasoning": verdict_obj.reasoning,
            "confidence": verdict_obj.confidence,
        }

        # Standardized blocking: determined by the judge's content-based
        # classification, independent of each provider's native detection.
        # TP or FP means the judge considers the response a block/refusal.
        result.standardized_blocked = verdict_obj.classification in ("TP", "FP")

        return result_to_dict(case, provider.name, result, verdict)


async def run_configuration(
    provider: BaseProvider,
    cases: list[TestCase],
    judge: JudgeEvaluator,
    output_path: Path,
) -> list[dict]:
    """Run all test cases through one provider and save results."""
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENCY)
    print(f"\n{'='*60}")
    print(f"  Running configuration: {provider.name}")
    print(f"  Test cases: {len(cases)}")
    print(f"  Max concurrency: {config.MAX_CONCURRENCY}")
    print(f"{'='*60}\n")

    tasks = [run_single(semaphore, provider, case, judge) for case in cases]
    results = await tqdm_asyncio.gather(*tasks, desc=provider.name)

    # Write results to JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    print(f"\n  ✓ Results saved to {output_path}")
    return results


# ── Provider factory ──────────────────────────────────────────────────────────
PROVIDERS: dict[str, type[BaseProvider]] = {
    "control": ControlProvider,
    "guardrails_ai": GuardrailsAIProvider,
    "nemo_guardrails": NeMoGuardrailsProvider,
}


def build_provider(name: str) -> BaseProvider:
    cls = PROVIDERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown configuration: {name}. Choose from {list(PROVIDERS)}")
    return cls()


# ── Main ──────────────────────────────────────────────────────────────────────
async def main(
    configs: list[str],
    dry_run: bool = False,
    repetitions: int = 1,
    seed: int = 42,
):
    # Ensure output directory exists
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets
    cases = load_all_datasets()
    if not cases:
        print("[ERROR] No test cases loaded. Run `python datasets/build_dataset.py` first.")
        sys.exit(1)

    if dry_run:
        print(f"[DRY RUN] Would process {len(cases)} cases × {len(configs)} configs × {repetitions} rep(s).")
        print(f"[DRY RUN] Configs: {configs}")
        for c in cases[:3]:
            print(f"  Sample: [{c.category}] {c.prompt[:80]}...")
        return

    judge = JudgeEvaluator()
    rng = random.Random(seed)

    for rep in range(1, repetitions + 1):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rep_label = f"rep{rep}" if repetitions > 1 else ""

        # Randomize configuration order each repetition to mitigate
        # time-of-day confounds on API latency.
        run_order = list(configs)
        rng.shuffle(run_order)

        if repetitions > 1:
            print(f"\n{'#'*60}")
            print(f"  Repetition {rep}/{repetitions}  —  config order: {run_order}")
            print(f"{'#'*60}")

        all_results: dict[str, list[dict]] = {}

        for cfg_name in run_order:
            provider = build_provider(cfg_name)
            suffix = f"_{rep_label}" if rep_label else ""
            output_path = config.RESULTS_DIR / f"{cfg_name}_{timestamp}{suffix}.jsonl"
            results = await run_configuration(provider, cases, judge, output_path)
            all_results[cfg_name] = results

        # Write combined summary for this repetition
        suffix = f"_{rep_label}" if rep_label else ""
        summary_path = config.RESULTS_DIR / f"summary_{timestamp}{suffix}.json"
        summary = {
            "timestamp": timestamp,
            "repetition": rep,
            "total_repetitions": repetitions,
            "config_run_order": run_order,
            "total_cases": len(cases),
            "configurations": run_order,
            "files": {
                cfg: f"{cfg}_{timestamp}{suffix}.jsonl" for cfg in run_order
            },
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Experiment complete!  ({repetitions} repetition(s))")
    print(f"  Results directory: {config.RESULTS_DIR}")
    print(f"{'='*60}")
    print(f"\nNext step: run  python analysis.py  to compute metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safety Tax Experiment Runner")
    parser.add_argument(
        "--config",
        nargs="+",
        default=config.CONFIGURATIONS,
        choices=config.CONFIGURATIONS,
        help="Which configurations to run (default: all three).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and print plan without making API calls.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of full experiment repetitions (randomized config order each time). "
             "Use 3+ to report mean +/- std across runs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for config order shuffling (default: 42).",
    )
    args = parser.parse_args()

    asyncio.run(main(
        configs=args.config,
        dry_run=args.dry_run,
        repetitions=args.repetitions,
        seed=args.seed,
    ))
