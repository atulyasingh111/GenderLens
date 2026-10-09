import subprocess
import sys


STEPS = [
    ["python", "src/embedding_bias.py"],
    ["python", "src/debias.py"],
    ["python", "src/statistics.py"],
    ["python", "src/semantic_preservation.py"],
    ["python", "src/plot_comparison.py"],
]


def run_step(command):
    print("\n" + "=" * 60)
    print("Running:", " ".join(command))
    print("=" * 60)

    result = subprocess.run(command)

    if result.returncode != 0:
        print("\nPipeline stopped because a step failed.")
        sys.exit(result.returncode)


print("Starting GenderLens pipeline...")

for step in STEPS:
    run_step(step)

print("\n" + "=" * 60)
print("GenderLens pipeline completed successfully.")
print("=" * 60)
