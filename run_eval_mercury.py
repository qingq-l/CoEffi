#To change model,please view mercury-main/src/evaluator.py
import os
import sys

MERCURY_ROOT = "path/to/mercury"
if MERCURY_ROOT not in sys.path:
    sys.path.append(MERCURY_ROOT)

os.environ["PYTHONPATH"] = f"{MERCURY_ROOT}:{os.environ.get('PYTHONPATH', '')}"

from src import evaluator as Evaluator

CURRENT_MODEL = "path/to/model"
RUN_NAME = "run--mercury"
DO_GENERATE = True
DO_EVALUATE = True
NUM_SAMPLES = 1

def main():
    print(f"Initializing evaluator, loading model: {CURRENT_MODEL}")
    evaluator = Evaluator.DistributeWiseEvaluator(
        model_name_or_path=CURRENT_MODEL,
        do_generate=DO_GENERATE
    )

    if DO_GENERATE:
        print("\n[Phase 1] Generating code samples...")
        evaluator.generate(num_samples_per_task=NUM_SAMPLES)
    else:
        print("\n[Phase 1] Skipping generation, using existing samples...")

    if DO_EVALUATE:
        print("\n[Phase 2] Running sandbox evaluation...")
        evaluator.evaluate(num_samples_per_task=NUM_SAMPLES)

    print("\nEvaluation pipeline completed.")

if __name__ == "__main__":
    main()