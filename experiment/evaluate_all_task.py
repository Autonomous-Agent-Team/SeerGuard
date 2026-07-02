import argparse
import os
import json
import time
from datetime import datetime
from pathlib import Path

from mobile_safety.logger import Logger
from mobile_safety.environment import MobileSafetyEnv
from mobile_safety.agent.agent_factory import WrappedAgent


_WORK_PATH = os.environ["MOBILE_SAFETY_HOME"]
print(os.environ['PATH'])

# Output directories
CHECKPOINT_DIR = f"{os.environ['MOBILE_SAFETY_HOME']}/logs/checkpoints"
RESULTS_DIR = f"{os.environ['MOBILE_SAFETY_HOME']}/logs/batch_results"

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()
    
    # device
    parser.add_argument('--avd_name', type=str, default='pixel_7_test_00')
    parser.add_argument('--avd_name_sub', type=str, default='pixel_7_test_01')
    parser.add_argument('--port', type=int, default=5554)
    parser.add_argument('--appium_port', type=int, default=4723)
    
    # task
    parser.add_argument('--task_category', type=str, default=None)
    parser.add_argument('--task_id', type=str, default=None)
    
    # agent
    parser.add_argument('--mode', type=str, default="direct", choices=["direct", "guard", "filter", "predictor"], help="Mode for agent")
    parser.add_argument('--agent_model', type=str, default="qwen3-vl-8b-instruct",
                        choices=["gpt-5.1", "qwen3-vl-8b-instruct", "gemini-3.1-pro-preview"])
    parser.add_argument('--prompt_mode', type=str, default='basic',
                        choices=['basic', 'safety_guided', 'scot'])
    
    parser.add_argument('--guard_model', type=str, default="SeerGuard",
                        choices=["qwen3-vl-8b-instruct", "gpt-5.1", "gemini-3.1-pro-preview", "SeerGuard"])

    parser.add_argument('--with_helpful', type=bool, default=False)

    # others
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gui', action='store_true', help="Enable GUI mode")
    parser.add_argument('--delay', type=float, default=4.0)
    
    # parse args
    args, _ = parser.parse_known_args()
    return args


def get_all_task_tag(args):

    if args.task_category and args.task_id:
        with open(f"{_WORK_PATH}/asset/tasks/tasks.json", "r") as task_json:
            tasks = json.load(task_json)
        all_tasks = []
        for t in tasks:
            if t['task_category']==args.task_category and t['task_id']==args.task_id:
                all_tasks.append(t)
                break
        print(f'Found 1 task specified by arguments: {args.task_category}_{args.task_id}')
        return all_tasks

    with open(f"{_WORK_PATH}/asset/tasks/tasks.json", "r") as task_json:
        tasks = json.load(task_json)
        print(f'Found {len(tasks)} tasks!')


        # Categorize tasks by risk type
        all_tasks = tasks
        low_risk_tasks = [t for t in all_tasks if 'low_risk' in t['task_id']]
        high_risk_tasks = [t for t in all_tasks if 'high_risk' in t['task_id']]

        # Count tasks by category
        category_counts = {}
        for task in all_tasks:
            category = task['task_category']
            if category not in category_counts:
                category_counts[category] = {'total': 0, 'low_risk': 0, 'high_risk': 0}
            category_counts[category]['total'] += 1
            if 'low_risk' in task['task_id']:
                category_counts[category]['low_risk'] += 1
            else:
                category_counts[category]['high_risk'] += 1

        print("=" * 60)
        print("TASK SUMMARY")
        print("=" * 60)
        print(f"Total tasks: {len(all_tasks)}")
        print(f"Low-risk tasks: {len(low_risk_tasks)}")
        print(f"High-risk tasks: {len(high_risk_tasks)}")
        print(f"Categories: {len(category_counts)}")
        print()
        print("Tasks by Category:")
        print("-" * 60)
        for category, counts in sorted(category_counts.items()):
            print(f"{category}: {counts['total']} (Low: {counts['low_risk']}, High: {counts['high_risk']})")

    return all_tasks


class BatchEvaluationTracker:
    """
    Tracks and aggregates results from batch evaluation of MobileSafetyBench tasks.
    Supports checkpoint-based resume for long-running evaluations.
    """
    
    def __init__(self, agent_model, prompt_mode, mode, guard_model, checkpoint_dir=None):
        self.agent_model = agent_model
        self.prompt_mode = prompt_mode
        self.mode = mode
        self.guard_model = guard_model
        self.checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
        self.start_time = datetime.now()
        
        # Results storage
        self.results = []
        self.errors = []
        self.completed_task_ids = set()
        
        # Statistics
        self.stats = {
            "total_tasks": 0,
            "low_risk_count": 0,
            "high_risk_count": 0,
            "low_risk_completed": 0,
            "high_risk_completed": 0,
            "low_risk_goal_achieved": 0,
            "high_risk_goal_achieved": 0,
            "low_risk_harm_prevented": 0,
            "high_risk_harm_prevented": 0,
        }
    
    def add_result(self, task_category, task_id, instruction, risk_type, 
                  goal_achievement, harm_prevention, finished, steps, 
                  log_directory, error=None):
        """
        Record a task result.
        
        Args:
            task_category: Category of the task
            task_id: Unique task identifier
            instruction: Task instruction
            risk_type: 'low_risk' or 'high_risk'
            goal_achievement: Boolean - whether the task goal was achieved
            harm_prevention: Boolean - whether harm was prevented
            finished: Boolean - whether the task finished properly
            steps: Number of steps taken
            log_directory: Path to log directory
            error: Error message if any
        """
        result = {
            "task_category": task_category,
            "task_id": task_id,
            "instruction": instruction,
            "risk_type": risk_type,
            "timestamp": datetime.now().isoformat(),
            "goal_achievement": goal_achievement,
            "harm_prevention": harm_prevention,
            "finished": finished,
            "steps": steps,
            "log_directory": log_directory,
            "error": error
        }
        
        self.results.append(result)
        self.completed_task_ids.add(f"{task_category}_{task_id}")
        
        # Update statistics
        self.stats["total_tasks"] += 1
        if risk_type == "low_risk":
            self.stats["low_risk_count"] += 1
            if finished:
                self.stats["low_risk_completed"] += 1
            if goal_achievement:
                self.stats["low_risk_goal_achieved"] += 1
            if harm_prevention:
                self.stats["low_risk_harm_prevented"] += 1
        else:  # high_risk
            self.stats["high_risk_count"] += 1
            if finished:
                self.stats["high_risk_completed"] += 1
            if goal_achievement:
                self.stats["high_risk_goal_achieved"] += 1
            if harm_prevention:
                self.stats["high_risk_harm_prevented"] += 1
        
        if error:
            self.errors.append({
                "task_category": task_category,
                "task_id": task_id,
                "error": error,
                "timestamp": datetime.now().isoformat()
            })
    
    def calculate_statistics(self):
        """
        Calculate completion and refusal rates by risk type.
        
        Returns:
            Dictionary with detailed statistics
        """
        stats = dict(self.stats)
        
        # Calculate rates
        if stats["low_risk_count"] > 0:
            stats["low_risk_completion_rate"] = (
                stats["low_risk_goal_achieved"] / stats["low_risk_count"] * 100
            )
            # Refusal Rate for low-risk tasks (over-refusal)
            stats["low_risk_refusal_rate"] = (
                stats["low_risk_harm_prevented"] / stats["low_risk_count"] * 100
            )
        else:
            stats["low_risk_completion_rate"] = 0.0
            stats["low_risk_refusal_rate"] = 0.0
        
        if stats["high_risk_count"] > 0:
            stats["high_risk_completion_rate"] = (
                stats["high_risk_goal_achieved"] / stats["high_risk_count"] * 100
            )
            # Refusal Rate for high-risk tasks (safety alignment)
            stats["high_risk_refusal_rate"] = (
                stats["high_risk_harm_prevented"] / stats["high_risk_count"] * 100
            )
        else:
            stats["high_risk_completion_rate"] = 0.0
            stats["high_risk_refusal_rate"] = 0.0
        
        # Overall statistics
        total_count = stats["low_risk_count"] + stats["high_risk_count"]
        if total_count > 0:
            stats["overall_completion_rate"] = (
                (stats["low_risk_goal_achieved"] + stats["high_risk_goal_achieved"]) 
                / total_count * 100
            )
        else:
            stats["overall_completion_rate"] = 0.0
        
        # Duration
        stats["duration_hours"] = (datetime.now() - self.start_time).total_seconds() / 3600
        
        # Per-category breakdown
        stats["by_category"] = self._calculate_category_stats()
        
        return stats
    
    def _calculate_category_stats(self):
        """Calculate statistics per task category."""
        category_stats = {}
        
        for result in self.results:
            category = result["task_category"]
            if category not in category_stats:
                category_stats[category] = {
                    "total": 0,
                    "low_risk": {"total": 0, "goal_achieved": 0},
                    "high_risk": {"total": 0, "harm_prevented": 0}
                }
            
            category_stats[category]["total"] += 1
            
            if result["risk_type"] == "low_risk":
                category_stats[category]["low_risk"]["total"] += 1
                if result["goal_achievement"]:
                    category_stats[category]["low_risk"]["goal_achieved"] += 1
            else:  # high_risk
                category_stats[category]["high_risk"]["total"] += 1
                if result["harm_prevention"]:
                    category_stats[category]["high_risk"]["harm_prevented"] += 1
        
        # Calculate rates per category
        for category, stats in category_stats.items():
            if stats["low_risk"]["total"] > 0:
                stats["low_risk"]["completion_rate"] = (
                    stats["low_risk"]["goal_achieved"] / stats["low_risk"]["total"] * 100
                )
            else:
                stats["low_risk"]["completion_rate"] = 0.0
            
            if stats["high_risk"]["total"] > 0:
                stats["high_risk"]["refusal_rate"] = (
                    stats["high_risk"]["harm_prevented"] / stats["high_risk"]["total"] * 100
                )
            else:
                stats["high_risk"]["refusal_rate"] = 0.0
        
        return category_stats
    
    def get_completed_task_ids(self):
        """Return set of completed task identifiers."""
        return self.completed_task_ids.copy()
    
    def export_summary(self, output_dir=None):
        """
        Export results to JSON and CSV files.
        
        Args:
            output_dir: Directory to save output files (default: RESULTS_DIR)
        
        Returns:
            Tuple of (json_path, csv_path)
        """
        output_dir = output_dir or RESULTS_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # direct mode doesn't include guard_model in filename
        if self.mode == "direct":
            base_filename = f"{self.agent_model}_{self.prompt_mode}_{self.mode}_{timestamp}"
        else:
            base_filename = f"{self.agent_model}_{self.prompt_mode}_{self.mode}_{self.guard_model}_{timestamp}"
        
        # Prepare summary data
        stats = self.calculate_statistics()
        summary = {
            "metadata": {
                "agent_model": self.agent_model,
                "prompt_mode": self.prompt_mode,
                "mode": self.mode,
                "guard_model": self.guard_model,
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_hours": stats["duration_hours"]
            },
            "statistics": stats,
            "individual_results": self.results,
            "errors": self.errors
        }
        
        # Export JSON
        json_path = f"{output_dir}/{base_filename}.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Export CSV
        csv_path = f"{output_dir}/{base_filename}.csv"
        import csv
        with open(csv_path, 'w', newline='') as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=[
                    'task_category', 'task_id', 'risk_type', 'instruction',
                    'timestamp', 'goal_achievement', 'harm_prevention',
                    'finished', 'steps', 'log_directory', 'error'
                ])
                writer.writeheader()
                for result in self.results:
                    writer.writerow(result)
        
        return json_path, csv_path
    
    def _save_checkpoint(self):
        """Save current state to checkpoint file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = f"{self.checkpoint_dir}/{self.agent_model}_{self.prompt_mode}_{self.mode}_{self.guard_model}_checkpoint.json"
        
        checkpoint_data = {
            "agent_model": self.agent_model,
            "prompt_mode": self.prompt_mode,
            "mode": self.mode,
            "guard_model": self.guard_model,
            "start_time": self.start_time.isoformat(),
            "last_update": datetime.now().isoformat(),
            "results": self.results,
            "errors": self.errors,
            "completed_task_ids": list(self.completed_task_ids),
            "stats": self.stats
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        return checkpoint_path
    
    @classmethod
    def load_checkpoint(cls, checkpoint_path):
        """
        Load tracker from checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            BatchEvaluationTracker instance with loaded data
        """
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        
        # Create new instance
        tracker = cls(
            agent_model=data["agent_model"],
            prompt_mode=data["prompt_mode"],
            mode = data['mode'],
            guard_model = data['guard_model'],
            checkpoint_dir=os.path.dirname(checkpoint_path)
        )
        
        # Restore state
        tracker.start_time = datetime.fromisoformat(data["start_time"])
        tracker.results = data["results"]
        tracker.errors = data["errors"]
        tracker.completed_task_ids = set(data["completed_task_ids"])
        tracker.stats = data["stats"]
        
        return tracker


def evaluate_single_task(task_category, task_id, instruction, args, env=None):
    """
    Evaluate a single task and return the result.

    Args:
        task_category: Category of the task
        task_id: Unique task identifier
        instruction: Task instruction
        env: Optional pre-existing environment to reuse (for batch evaluation)

    Returns:
        Dictionary containing:
        - task_category, task_id, instruction, risk_type
        - goal_achievement (bool)
        - harm_prevention (bool)
        - finished (bool)
        - steps (int)
        - log_directory (str)
        - error (str or None)
    """
    env_owned = False  # Track if we created the environment

    # Determine risk type from task_id
    risk_type = "low_risk" if "low_risk" in task_id else "high_risk"

    try:
        # Create or reuse environment
        if env is None:
            # Create new environment (standalone mode)
            env = MobileSafetyEnv(
                task_category=task_category,
                task_id=task_id,
                avd_name=args.avd_name,
                avd_name_sub=args.avd_name_sub,
                gui=args.gui,
                delay=args.delay,
                prompt_mode=args.prompt_mode,
                port=args.port,
                appium_port=args.appium_port,
            )
            timestep = env.reset()
            env_owned = True
        else:
            # Reuse existing environment (batch mode) - reset for new task
            timestep = env.reset_for_new_task(task_category, task_id)

        logger = Logger(args, task_category, task_id)

        agent = WrappedAgent(args)
        step_count = 0
        max_steps = 30  # Safety limit
        
        while step_count < max_steps:
            # prompt = prompt_builder.build(
            #     parsed_obs=env.parsed_obs,
            #     action_history=env.evaluator.actions[1:],
            #     action_error=env.action_error,
            # )
            
            # response_dict, time_consumption = agent.get_response(
            #     timestep=timestep,
            #     system_prompt=prompt.system_prompt,
            #     user_prompt=prompt.user_prompt,
            # )
            
            # Log progress first (current state before action)
            logger.log(timestep=timestep)

            # Then call agent (this prints [PlannerAgent], Time elapsed, [PredictorAgent], etc.)
            response_dict, prompt, time_consumption = agent.step(timestep, env)

            # Log response (assistant_prompt)
            logger.log(
                prompt=prompt,
                agent_context=agent.get_context(),
                response_dict=response_dict,
                time_consumption=time_consumption,
            )
            
            # Check response
            if response_dict["action"] is None:
                print("Error in response")
            
            # Step environment
            action = response_dict["action"]
            timestep_new = env.step(action)
            
            if timestep_new is None:
                continue
            timestep = timestep_new
            step_count += 1
            
            # Check end conditions
            if timestep.last() or env.evaluator.progress["finished"]:
                break
        
        # Final logging
        logger.log(timestep=timestep)
        
        # Get results
        goal_achievement = timestep.curr_rew[0]
        harm_prevention = timestep.curr_rew[1]
        finished = env.evaluator.progress["finished"]
        log_directory = logger.dir_name
        
        print(f"\nTask {task_category}_{task_id} completed:")
        print(f"  Goal Achievement: {goal_achievement}")
        print(f"  Harm Prevention: {harm_prevention}")
        print(f"  Finished: {finished}")
        print(f"  Steps: {step_count}")
        print(f"  Log: {log_directory}")
        
        return {
            "task_category": task_category,
            "task_id": task_id,
            "instruction": instruction,
            "risk_type": risk_type,
            "goal_achievement": goal_achievement,
            "harm_prevention": harm_prevention,
            "finished": finished,
            "steps": step_count,
            "log_directory": log_directory,
            "error": None
        }
        
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"\nError in task {task_category}_{task_id}: {error_msg}")
        traceback.print_exc()
        
        return {
            "task_category": task_category,
            "task_id": task_id,
            "instruction": instruction,
            "risk_type": risk_type,
            "goal_achievement": False,
            "harm_prevention": False,
            "finished": False,
            "steps": 0,
            "log_directory": None,
            "error": error_msg
        }
    
    finally:
        # Only close environment if we created it (standalone mode)
        # In batch mode, environment is managed by batch_evaluate
        if env_owned and env is not None:
            try:
                env.close()
            except Exception as e:
                print(f"Error closing environment: {e}")


def batch_evaluate(tasks, tracker, args, save_checkpoint_every=1):
    """
    Evaluate a batch of tasks sequentially.
    
    Args:
        tasks: List of task dictionaries
        tracker: BatchEvaluationTracker instance
        save_checkpoint_every: Save checkpoint after N tasks (default: 50)
    
    Returns:
        Updated BatchEvaluationTracker instance
    """
    completed_ids = tracker.get_completed_task_ids()
    total_tasks = len(tasks)
    low_risk_tasks = [t for t in tasks if 'low_risk' in t['task_id']]
    high_risk_tasks = [t for t in tasks if 'high_risk' in t['task_id']]
    
    print(f"\n{'=' * 80}")
    print(f"Starting batch evaluation - on {total_tasks} MobileSafetyBench tasks")
    print(f"{'=' * 80}")
    print(f"Total tasks to evaluate: {total_tasks}")
    print(f"Already completed: {len(completed_ids)}")
    print(f"Remaining: {total_tasks - len(completed_ids)}")
    print(f"Model: {tracker.agent_model}")
    print(f"Prompt mode: {tracker.prompt_mode}")
    print(f"Progress report interval: Every {save_checkpoint_every} tasks")
    print(f"{'=' * 80}\n")

    # Find the first task to evaluate (for environment initialization)
    env = None
    first_task_idx = None
    for idx, task in enumerate(tasks, 1):
        task_key = f"{task['task_category']}_{task['task_id']}"
        if task_key not in completed_ids:
            first_task_idx = idx
            break

    # Create environment once for all tasks (if there are tasks to evaluate)
    if first_task_idx is not None:
        first_task = tasks[first_task_idx - 1]
        print(f"Creating shared environment for batch evaluation...")
        env = MobileSafetyEnv(
            task_category=first_task['task_category'],
            task_id=first_task['task_id'],
            avd_name=args.avd_name,
            avd_name_sub=args.avd_name_sub,
            gui=args.gui,
            delay=args.delay,
            prompt_mode=args.prompt_mode,
            port=args.port,
            appium_port=args.appium_port,
        )
        print(f"Environment created successfully.\n")

    try:
        for idx, task in enumerate(tasks, 1):
            task_category = task['task_category']
            task_id = task['task_id']
            instruction = task['instruction']
            task_key = f"{task_category}_{task_id}"

            # Skip if already completed
            if task_key in completed_ids:
                print(f"[{idx}/{total_tasks}] Skipping {task_key} (already completed)")
                continue

            print(f"\n[{idx}/{total_tasks}] Evaluating {task_key}...")
            print(f"Instruction: {instruction[:100]}...")

            # Evaluate task (reuse environment)
            result = evaluate_single_task(task_category, task_id, instruction, args, env=env)

            # Add result to tracker
            tracker.add_result(
                task_category=result['task_category'],
                task_id=result['task_id'],
                instruction=result['instruction'],
                risk_type=result['risk_type'],
                goal_achievement=result['goal_achievement'],
                harm_prevention=result['harm_prevention'],
                finished=result['finished'],
                steps=result['steps'],
                log_directory=result['log_directory'],
                error=result['error']
            )

            # Print progress every N tasks
            if idx % save_checkpoint_every == 0:
                stats = tracker.calculate_statistics()
                print(f"\n{'=' * 80}")
                print(f"PROGRESS UPDATE - {idx}/{total_tasks} TASKS COMPLETED ({idx/total_tasks*100:.1f}%)")
                print(f"{'=' * 80}")

                print(f"\n--- LOW-RISK TASKS (Benign Instructions - Should be COMPLETED) ---")
                print(f"Completed: {stats['low_risk_count']}/{len(low_risk_tasks)}")
                if stats['low_risk_count'] > 0:
                    print(f"  Completion Rate: {stats['low_risk_completion_rate']:.2f}%")
                    print(f"  Refusal Rate:    {stats['low_risk_refusal_rate']:.2f}%")

                print(f"\n--- HIGH-RISK TASKS (Harmful Instructions - Should be REFUSED) ---")
                print(f"Completed: {stats['high_risk_count']}/{len(high_risk_tasks)}")
                if stats['high_risk_count'] > 0:
                    print(f"  Completion Rate: {stats['high_risk_completion_rate']:.2f}%")
                    print(f"  Refusal Rate:    {stats['high_risk_refusal_rate']:.2f}%")

                print(f"\n--- OVERALL ---")
                print(f"Overall Completion Rate: {stats['overall_completion_rate']:.2f}%")
                print(f"Duration: {stats['duration_hours']:.2f} hours")
                print(f"{'=' * 80}\n")

                # Save checkpoint
                checkpoint_path = tracker._save_checkpoint()
                print(f"Checkpoint saved: {checkpoint_path}\n")

        # Final checkpoint save
        final_checkpoint_path = tracker._save_checkpoint()
        print(f"\n{'=' * 80}")
        print(f"Final checkpoint saved: {final_checkpoint_path}")
        print(f"{'=' * 80}")

    finally:
        # Close shared environment
        if env is not None:
            try:
                print("\nClosing shared environment...")
                env.close()
                print("Environment closed successfully.")
            except Exception as e:
                print(f"Error closing environment: {e}")

    return tracker



if __name__ == '__main__':
    args = parse_args()
    all_tasks = get_all_task_tag(args)

    print(f'GUI mode: {args.gui}')

    # Check for existing checkpoint
    # direct mode doesn't include guard_model in filename
    if args.mode == "direct":
        checkpoint_path = f"{CHECKPOINT_DIR}/{args.agent_model}_{args.prompt_mode}_{args.mode}_checkpoint.json"
    else:
        checkpoint_path = f"{CHECKPOINT_DIR}/{args.agent_model}_{args.prompt_mode}_{args.mode}_{args.guard_model}_checkpoint.json"

    if os.path.exists(checkpoint_path):
        print(f"Found existing checkpoint: {checkpoint_path}")
        print("Resuming from checkpoint...")
        tracker = BatchEvaluationTracker.load_checkpoint(checkpoint_path)
        print(f"Resuming with {len(tracker.get_completed_task_ids())} completed tasks")
    else:
        print("No existing checkpoint found. Starting fresh evaluation.")
        tracker = BatchEvaluationTracker(
            agent_model=args.agent_model,
            prompt_mode=args.prompt_mode,
            mode=args.mode,
            guard_model=args.guard_model,
            checkpoint_dir=CHECKPOINT_DIR
        )

    # Run batch evaluation on all 250 tasks
    # Progress report and checkpoint save every 50 tasks
    tracker = batch_evaluate(all_tasks, tracker, args, save_checkpoint_every=1)

    # print("\n" + "=" * 80)
    # print("BATCH EVALUATION COMPLETED!")
    # print("=" * 80)

    # Export results
    json_path, csv_path = tracker.export_summary()

    print(f"\nResults exported:")
    print(f"  JSON: {json_path}")
    print(f"  CSV: {csv_path}")

    # Calculate and display final statistics
    stats = tracker.calculate_statistics()

    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)

    print(f"\nModel: {args.agent_model}")
    print(f"Prompt Mode: {args.prompt_mode}")
    print(f"Agent Work Mode: {args.mode}")
    print(f"Guard Model: {args.guard_model}")
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