import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

INPUT_DIR = "/home/kiemax/Helmholtz/projects/proteinTPM/noripTPM/data/sabdab/interfaces"
SCRIPT_PATH = "scripts/stack_features_cartesian.py"
MAX_WORKERS = 4


def run(shard):
    cmd = [
        "python",
        SCRIPT_PATH,
        "--shard",
        str(shard),  # Convert Path object to string
        "--saveTo",
        str(shard),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return (True, shard)
    except subprocess.CalledProcessError as e:
        return (False, shard, e.stderr.decode())


if __name__ == "__main__":
    # Get all shard directories from INPUT_DIR
    shards = [f for f in Path(INPUT_DIR).iterdir() if f.is_dir()]
    failures = []

    # Run the tasks in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run, shard) for shard in shards]

        for future in as_completed(futures):
            result = future.result()
            if result[0]:
                print(f"{result[1]} processed")
            else:
                print(f"{result[1]} failed")
                failures.append(result[2])  # Log error message

    if failures:
        print("Some shards failed processing:")
        for error in failures:
            print(error)
