"""
Experiment utilities for MobileSafetyBench.

This module provides functions for:
- Cleaning up emulator/appium processes
- Logging output to files
- Running evaluation experiments
- Loading experiment results from batch results and checkpoints
- Plotting comparison charts for experiment analysis
"""

import os
import sys
import subprocess
import time
import json
from types import SimpleNamespace
from datetime import datetime
from contextlib import contextmanager

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import glob

# Configure matplotlib for text display
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from experiment.evaluate_all_task import (
    get_all_task_tag,
    BatchEvaluationTracker,
    batch_evaluate,
    CHECKPOINT_DIR,
    RESULTS_DIR,
)

# Project path setup
project_path = os.environ.get("MOBILE_SAFETY_HOME")
os.environ["MOBILE_SAFETY_HOME"] = project_path
os.chdir(project_path)
if project_path not in sys.path:
    sys.path.insert(0, project_path)


# ============================================================================
# Setup and experiment running utilities
# ============================================================================

def cleanup():
    """Kill leftover emulator / appium processes (mirrors run_eval.sh cleanup)."""
    print("\n" + "=" * 40)
    print("Cleaning up resources...")
    print("=" * 40)

    for proc in ["qemu-system-aarch64-headless", "emulator", "netsimd", "appium", "node"]:
        subprocess.run(["pkill", "-9", proc], capture_output=True)

    subprocess.run(["adb", "devices", "emu", "kill"], capture_output=True)

    print("Waiting for processes to exit...")
    time.sleep(5)

    result = subprocess.run(["adb", "devices"], capture_output=True, text=True)
    remaining = sum(1 for line in result.stdout.splitlines() if "emulator" in line)
    if remaining > 0:
        print(f"Warning: {remaining} emulator(s) still detected")
    else:
        print("All emulators cleaned up successfully")
    print("Cleanup complete")
    print("=" * 40 + "\n")


class _TeeStream:
    """Write to both the original stream and a log file."""

    def __init__(self, original, log_file):
        self.original = original
        self.log_file = log_file

    def write(self, data):
        self.original.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.original.flush()
        self.log_file.flush()

    # Forward everything else to the original stream
    def __getattr__(self, name):
        return getattr(self.original, name)


@contextmanager
def tee_output(log_path):
    """Context manager: mirror stdout/stderr to *log_path*."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(old_stdout, log_file)
    sys.stderr = _TeeStream(old_stderr, log_file)
    try:
        yield log_path
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_file.close()


def run_experiment(agent_model, mode, guard_model, prompt_mode="basic"):
    """Run one full evaluation experiment with cleanup, checkpoint resume, and logging."""

    # 1. Cleanup
    cleanup()

    # 2. Build args (mirrors parse_args defaults)
    args = SimpleNamespace(
        avd_name="pixel_7_test_00",
        avd_name_sub="pixel_7_test_01",
        port=5554,
        appium_port=4723,
        task_category=None,
        task_id=None,
        mode=mode,
        agent_model=agent_model,
        prompt_mode=prompt_mode,
        guard_model=guard_model,
        with_helpful=False,
        seed=42,
        gui=False,
        delay=5.0,
    )

    # Print args
    print("\n" + "=" * 50)
    print("Experiment Configuration:")
    print("=" * 50)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 50 + "\n")

    # 3. Task list
    all_tasks = get_all_task_tag(args)

    # 4. Checkpoint resume
    checkpoint_path = f"{CHECKPOINT_DIR}/{agent_model}_{prompt_mode}_{mode}_{guard_model}_checkpoint.json"
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        tracker = BatchEvaluationTracker.load_checkpoint(checkpoint_path)
        print(f"  Already completed: {len(tracker.get_completed_task_ids())} tasks")
    else:
        print("No checkpoint found. Starting fresh.")
        tracker = BatchEvaluationTracker(
            agent_model=agent_model,
            prompt_mode=prompt_mode,
            mode=mode,
            guard_model=guard_model,
        )

    # 5. Log path
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"{project_path}/logs/notebook_logs/{agent_model}_{prompt_mode}_{mode}_{guard_model}_{ts}.log"

    # 6. Run
    print(f"\nExperiment: agent={agent_model}  mode={mode}  guard={guard_model}")
    print(f"Log file:   {log_path}\n")

    with tee_output(log_path):
        tracker = batch_evaluate(all_tasks, tracker, args)

    # 7. Export & summary
    json_path, csv_path = tracker.export_summary()
    stats = tracker.calculate_statistics()

    print(f"\nResults exported:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print(f"\n--- Final Stats ---")
    print(f"Total tasks:             {stats['total_tasks']}")
    print(f"Overall completion rate:  {stats['overall_completion_rate']:.2f}%")
    print(f"Low-risk  completion:     {stats['low_risk_completion_rate']:.2f}%")
    print(f"High-risk refusal:        {stats['high_risk_refusal_rate']:.2f}%")
    print(f"Duration:                 {stats['duration_hours']:.2f} hours")

    return tracker


def run_evaluation(model, mode="direct", guard_model="none", task_filter=None, prompt_mode="basic"):
    """Unified evaluation function with support for task filtering and checkpoint resume."""
    print(f"Starting {model} evaluation (mode={mode})...")

    # Parse task_filter
    task_category = None
    task_id = None

    if task_filter:
        if "/" in task_filter:
            # Single task: "category/task_id"
            task_category, task_id = task_filter.split("/", 1)
            print(f"Single task: {task_category}/{task_id}")
        else:
            # Category filter: "category"
            task_category = task_filter
            print(f"Category filter: {task_category}")

    # 1. Cleanup
    cleanup()

    # 2. Build args
    args = SimpleNamespace(
        avd_name="pixel_7_test_00",
        avd_name_sub="pixel_7_test_01",
        port=5554,
        appium_port=4723,
        task_category=task_category if task_id else None,  # get_all_task_tag needs both
        task_id=task_id,
        mode=mode,
        agent_model=model,
        prompt_mode=prompt_mode,
        guard_model=guard_model,
        with_helpful=False,
        seed=42,
        gui=False,
        delay=4.0,
    )

    # Print args
    print("\n" + "=" * 50)
    print("Experiment Configuration:")
    print("=" * 50)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 50 + "\n")

    # 3. Get task list
    if task_filter and "/" not in task_filter:
        # Category filter: manually load and filter
        with open(f"{project_path}/asset/tasks/tasks.json", "r") as f:
            all_tasks = json.load(f)
        tasks = [t for t in all_tasks if t['task_category'] == task_category]
        print(f"Found {len(tasks)} tasks in category '{task_category}'")
    else:
        # All tasks or single task
        tasks = get_all_task_tag(args)

    # 4. Checkpoint resume
    # direct mode doesn't include guard_model in filename
    if mode == "direct":
        checkpoint_name = f"{model}_{prompt_mode}_{mode}"
    else:
        checkpoint_name = f"{model}_{prompt_mode}_{mode}_{guard_model}"
    if task_filter:
        checkpoint_name += f"_{task_filter.replace('/', '_')}"
    checkpoint_path = f"{CHECKPOINT_DIR}/{checkpoint_name}_checkpoint.json"

    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint")
        tracker = BatchEvaluationTracker.load_checkpoint(checkpoint_path)
        print(f"   Already completed: {len(tracker.get_completed_task_ids())} tasks")
    else:
        tracker = BatchEvaluationTracker(
            agent_model=model,
            prompt_mode=prompt_mode,
            mode=mode,
            guard_model=guard_model,
        )

    # 5. Log path
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{model}_{prompt_mode}_{mode}"
    if task_filter:
        log_name += f"_{task_filter.replace('/', '_')}"
    log_path = f"{project_path}/logs/notebook_logs/{log_name}_{ts}.log"

    print(f"Log: {log_path}")

    # 6. Run
    with tee_output(log_path):
        tracker = batch_evaluate(tasks, tracker, args)

    # 7. Stats
    stats = tracker.calculate_statistics()

    # Export results
    json_path, csv_path = tracker.export_summary()
    print(f"\nResults exported:")
    print(f"  JSON: {json_path}")
    print(f"  CSV: {csv_path}")

    # Display detailed final statistics
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)

    print(f"\nModel: {tracker.agent_model}")
    print(f"Prompt Mode: {tracker.prompt_mode}")
    print(f"Agent Work Mode: {tracker.mode}")
    print(f"Guard Model: {tracker.guard_model}")
    print(f"Duration: {stats['duration_hours']:.2f} hours")

    print(f"\n--- Overall Results ---")
    print(f"Total Tasks: {stats['total_tasks']}")
    print(f"Overall Completion Rate: {stats['overall_completion_rate']:.2f}%")

    print(f"\n--- Low-Risk Tasks (Benign Instructions) ---")
    print(f"Total Low-Risk Tasks: {stats['low_risk_count']}")
    print(f"Completion Rate: {stats['low_risk_completion_rate']:.2f}%")
    print(f"Refusal Rate:    {stats['low_risk_refusal_rate']:.2f}%")

    print(f"\n--- High-Risk Tasks (Harmful Instructions) ---")
    print(f"Total High-Risk Tasks: {stats['high_risk_count']}")
    print(f"Completion Rate: {stats['high_risk_completion_rate']:.2f}%")
    print(f"Refusal Rate:    {stats['high_risk_refusal_rate']:.2f}%")

    print(f"\n--- Per-Category Breakdown ---")
    print(f"{'Category':<30} {'Total':>6} {'Low-Risk Completion':>20} {'High-Risk Refusal':>19}")
    print("-" * 80)
    for category, category_stats in sorted(stats['by_category'].items()):
        total = category_stats['total']
        low_rate = f"{category_stats['low_risk']['completion_rate']:.1f}%"
        high_rate = f"{category_stats['high_risk']['refusal_rate']:.1f}%"
        print(f"{category:<30} {total:>6} {low_rate:>20} {high_rate:>19}")

    print("\n" + "=" * 60)

    # Display tasks that didn't finish
    unfinished = [r for r in tracker.results if not r['finished']]

    if unfinished:
        print(f"\n{'=' * 60}")
        print(f"TASKS THAT DIDN'T FINISH ({len(unfinished)})")
        print(f"{'=' * 60}")
        for result in unfinished[:20]:  # Show first 20
            print(f"\n{result['task_category']}_{result['task_id']}:")
            print(f"  Risk Type: {result['risk_type']}")
            print(f"  Instruction: {result['instruction'][:80]}...")
            print(f"  Goal Achievement: {result['goal_achievement']}")
            print(f"  Harm Prevention: {result['harm_prevention']}")
            print(f"  Steps: {result['steps']}")
        if len(unfinished) > 20:
            print(f"\n... and {len(unfinished) - 20} more")
    else:
        print("\nAll tasks finished successfully!")

    # Display errors
    if tracker.errors:
        print(f"\n{'=' * 60}")
        print(f"TASKS WITH ERRORS ({len(tracker.errors)})")
        print(f"{'=' * 60}")
        for error in tracker.errors[:20]:  # Show first 20
            print(f"\n{error['task_category']}_{error['task_id']}:")
            print(f"  Error: {error['error']}")
            print(f"  Timestamp: {error['timestamp']}")
        if len(tracker.errors) > 20:
            print(f"\n... and {len(tracker.errors) - 20} more")
    else:
        print("\nNo errors encountered!")

    return tracker


# ============================================================================
# Results analysis utilities
# ============================================================================

def load_experiment_results(project_path=None):
    """Load experiment results from batch_results and checkpoints.

    Args:
        project_path: Path to the project root. If None, uses MOBILE_SAFETY_HOME env var.

    Returns:
        Tuple of (experiments dict, DataFrame with summary stats)
    """
    if project_path is None:
        project_path = os.environ.get("MOBILE_SAFETY_HOME")

    results_dir = f"{project_path}/logs/batch_results"
    checkpoint_dir = f"{project_path}/logs/checkpoints"

    # 1. First try to load complete results from batch_results
    experiments = {}

    batch_files = sorted(glob.glob(f"{results_dir}/*.json"))
    print(f"Found {len(batch_files)} batch result file(s) in {results_dir}/")

    for fpath in batch_files:
        fname = os.path.basename(fpath)
        with open(fpath) as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        agent = meta.get("agent_model", "unknown")
        prompt = meta.get("prompt_mode", "basic")
        mode = meta.get("mode", "direct")
        guard = meta.get("guard_model", "none")
        key = f"{agent}_{prompt}_{mode}_{guard}"
        experiments[key] = ("batch", data, fpath)

    # 2. For experiments without complete results, load from checkpoint
    checkpoint_files = sorted(glob.glob(f"{checkpoint_dir}/*_checkpoint.json"))
    print(f"Found {len(checkpoint_files)} checkpoint file(s) in {checkpoint_dir}/")

    for fpath in checkpoint_files:
        fname = os.path.basename(fpath)
        with open(fpath) as f:
            data = json.load(f)
        agent = data.get("agent_model", "unknown")
        prompt = data.get("prompt_mode", "basic")
        mode = data.get("mode", "direct")
        guard = data.get("guard_model", "none")
        key = f"{agent}_{prompt}_{mode}_{guard}"

        # Only use checkpoint if not in batch_results
        if key not in experiments:
            experiments[key] = ("checkpoint", data, fpath)

    print(f"Unique experiments: {len(experiments)}\n")

    # 3. Process data and build DataFrame
    rows = []
    for key, (source_type, data, fpath) in experiments.items():
        if source_type == "batch":
            meta = data.get("metadata", {})
            stats = data["statistics"]
            agent = meta.get("agent_model", "unknown")
            prompt = meta.get("prompt_mode", "basic")
            mode = meta.get("mode", "direct")
            guard = meta.get("guard_model", "none")
        else:  # checkpoint
            stats = data["stats"]
            agent = data.get("agent_model", "unknown")
            prompt = data.get("prompt_mode", "basic")
            mode = data.get("mode", "direct")
            guard = data.get("guard_model", "none")
            # Calculate rates from original counts
            lr_count = stats.get("low_risk_count", 1)
            hr_count = stats.get("high_risk_count", 1)
            stats["low_risk_completion_rate"] = stats.get("low_risk_goal_achieved", 0) / lr_count * 100 if lr_count > 0 else 0
            stats["high_risk_completion_rate"] = stats.get("high_risk_goal_achieved", 0) / hr_count * 100 if hr_count > 0 else 0
            stats["low_risk_refusal_rate"] = stats.get("low_risk_harm_prevented", 0) / lr_count * 100 if lr_count > 0 else 0
            stats["high_risk_refusal_rate"] = stats.get("high_risk_harm_prevented", 0) / hr_count * 100 if hr_count > 0 else 0

        # Generate label
        if mode == "direct":
            label = f"{agent}_{mode}"
        elif mode == "guard":
            label = f"{agent}_{guard}"
        else:
            label = f"{agent}_{mode}_{guard}"

        # If prompt_mode is scot, add to label
        if prompt == "scot":
            label = f"{label}_scot"

        rows.append({
            "Experiment": label,
            "Total": stats.get("total_tasks", 0),
            "LR Count": stats.get("low_risk_count", 0),
            "HR Count": stats.get("high_risk_count", 0),
            "LR Completion %": stats.get("low_risk_completion_rate", 0),
            "LR Refusal %": stats.get("low_risk_refusal_rate", 0),
            "HR Completion %": stats.get("high_risk_completion_rate", 0),
            "HR Refusal %": stats.get("high_risk_refusal_rate", 0),
        })
        print(f"  Loaded ({source_type}): {os.path.basename(fpath)} -> {label}")

    df = pd.DataFrame(rows)
    print("\n")

    return experiments, df


def plot_comparison_charts(df, experiments, project_path=None):
    """Plot side-by-side bar charts for experiment comparison.

    Args:
        df: DataFrame with experiment statistics
        experiments: Dict of experiment data (from load_experiment_results)
        project_path: Path to the project root. If None, uses MOBILE_SAFETY_HOME env var.
    """
    if project_path is None:
        project_path = os.environ.get("MOBILE_SAFETY_HOME")

    # Set Times New Roman font for all text
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'  # For math text

    labels = df["Experiment"].tolist()
    x = np.arange(len(labels))
    width = 0.25

    hr_comp = df["HR Completion %"].values
    lr_comp = df["LR Completion %"].values
    hr_ref = df["HR Refusal %"].values
    lr_ref = df["LR Refusal %"].values

    color_hr = '#D61C4E'   # red for high-risk
    color_lr = '#4466EE'   # blue for low-risk

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))

    # --- Left: Goal Achievement Rate ---
    ax1.bar(x - width/2, hr_comp, width, label='High-risk task', color=color_hr)
    ax1.bar(x + width/2, lr_comp, width, label='Low-risk task', color=color_lr)
    ax1.set_title('Goal Achievement Rate (%)', fontsize=24, pad=25)
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha='right', fontsize=16)
    ax1.grid(axis='x', visible=False)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    ax1.legend(fontsize=16, loc='upper right', frameon=True)
    # Value labels
    for xi, hv, lv in zip(x, hr_comp, lr_comp):
        ax1.text(xi - width/2, hv + 1, f'{hv:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax1.text(xi + width/2, lv + 1, f'{lv:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # --- Right: Refusal Rate ---
    ax2.bar(x - width/2, hr_ref, width, label='High-risk task', color=color_hr)
    ax2.bar(x + width/2, lr_ref, width, label='Low-risk task', color=color_lr)
    ax2.set_title('Refusal Rate (%)', fontsize=24, pad=25)
    ax2.set_ylim(0, 100)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha='right', fontsize=16)
    ax2.grid(axis='x', visible=False)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    ax2.legend(fontsize=16, loc='upper right', frameon=True)
    for xi, hv, lv in zip(x, hr_ref, lr_ref):
        ax2.text(xi - width/2, hv + 1, f'{hv:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax2.text(xi + width/2, lv + 1, f'{lv:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_path = f"{project_path}/logs/batch_results/experiment_comparison.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\nChart saved to {save_path}")
