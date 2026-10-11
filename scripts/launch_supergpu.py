"""
Example GPU launcher for training all shard models (machine-specific).

NOTE: This is provided as a reference for how the 36 shard models were trained
in parallel on our local GPU node. It assumes Singularity and a specific
container image path, and manages GPU locks under /tmp. It will need adaptation
for any other environment. The portable entry point is scripts/train_all_models.py.
"""

import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time

# --- USER SETTINGS ---
SIF_PATH = os.path.expanduser("~/Software/ml-gpu-env_v0.1.sif")
SCRIPT = "scripts/train_model.py"
INPUTS = [
    "Backbone_1",
    "Backbone_2",
    "Backbone_3",
    "Terminal_amino_1",
    "Terminal_amino_2",
    "Guanidino_1",
    "Guanidino_2",
    "Hydroxyl_1",
    "Hydroxyl_2",
    "Thiol_1",
    "Thiol_2",
    "Thioether_1",
    "Thioether_2",
    "Carboxyl_1",
    "Carboxyl_2",
    "Carboxamid_1",
    "Carboxamid_2",
    "Carboxamid_3",
    "Phe_1",
    "Phe_2",
    "Tyr_1",
    "Tyr_2",
    "Trp_1",
    "Trp_2",
    "Trp_3",
    "His_1",
    "His_2",
    "His_3",
    "Pro_1",
    "Pro_2",
    "Pro_3",
    "Gly_1",
    "Ala_1",
    "Leu_1",
    "Val_1",
    "Ile_1",
]


MAX_GPUS_TO_USE = 6
LOCK_DIR = "/tmp/gpu_locks"  # Use a shared path if multiple users/nodes contend
IDLE_UTIL_THRESH = 5  # ≤5% utilization counts as idle
IDLE_MEM_THRESH_MB = 500  # consider "free" if used memory < 500 MB
POLL_INTERVAL_SEC = 20


def run(cmd, **kw):
    return subprocess.check_output(cmd, text=True, **kw)


def list_gpus():
    out = run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    for line in out.strip().splitlines():
        idx, uuid, util, mem = [x.strip() for x in line.split(",")]
        gpus.append(
            {
                "index": int(idx),
                "uuid": uuid,
                "util": int(util),
                "mem": int(mem),
            }
        )
    return gpus


def gpu_busy_pids_by_uuid():
    """
    Map GPU UUID -> list of PIDs that have active compute contexts.
    If nvidia-smi compute-apps is empty, returns {}.
    """
    try:
        out = run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    except subprocess.CalledProcessError:
        # Some drivers don’t support compute-apps; fall back to utilization+mem
        return {}

    busy = {}
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        uuid, pid, _name, _mem = [x.strip() for x in line.split(",")]
        busy.setdefault(uuid, []).append(int(pid))
    return busy


def is_gpu_idle(g, busy_map):
    # If we can see active compute PIDs -> treat as busy
    if g["uuid"] in busy_map and busy_map[g["uuid"]]:
        return False
    # Otherwise use conservative thresholds
    return (g["util"] <= IDLE_UTIL_THRESH) and (g["mem"] < IDLE_MEM_THRESH_MB)


def try_lock_gpu(idx):
    path = os.path.join(LOCK_DIR, f"gpu{idx}.lock")
    try:
        os.makedirs(LOCK_DIR, exist_ok=True)
        os.mkdir(path)
        with open(os.path.join(path, "owner"), "w") as f:
            f.write(f"pid={os.getpid()}\n")
        return path
    except FileExistsError:
        return None


def unlock(path):
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


def find_and_lock_idle_gpus(target_count):
    locked = []
    try:
        while len(locked) < target_count:
            gpus = list_gpus()
            busy = gpu_busy_pids_by_uuid()
            # Prefer truly idle GPUs first
            candidates = [g for g in gpus if is_gpu_idle(g, busy)]
            # If not enough strictly idle, just consider unlocked ones (comment next line to be stricter)
            if len(candidates) < target_count:
                candidates = sorted(gpus, key=lambda g: (g["util"], g["mem"]))
            # Attempt to lock
            for g in candidates:
                if len(locked) >= target_count:
                    break
                lockpath = try_lock_gpu(g["index"])
                if lockpath:
                    # Re-check right after locking to be extra safe
                    g2 = [x for x in list_gpus() if x["index"] == g["index"]][0]
                    busy2 = gpu_busy_pids_by_uuid()
                    if is_gpu_idle(g2, busy2):
                        locked.append((g["index"], lockpath))
                    else:
                        unlock(lockpath)
            if len(locked) < target_count:
                # Didn’t get enough — wait and retry
                # Clean any partial locks we took this round
                for _, lp in locked:
                    unlock(lp)
                locked.clear()
                print(f"[launcher] Waiting for {target_count} idle GPUs...")
                time.sleep(POLL_INTERVAL_SEC)
        return locked
    except Exception:
        # On any unexpected error, clean locks and re-raise
        for _, lp in locked:
            unlock(lp)
        raise


def launch_worker(gpu_index, inp_queue):
    env = os.environ.copy()
    # Pin each worker to a single GPU
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    while True:
        try:
            inp = inp_queue.get_nowait()
        except queue.Empty:
            break
        cmd = ["singularity", "exec", "--nv", SIF_PATH, "python", SCRIPT, "--shard", inp]
        print(f"[GPU {gpu_index}] Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, env=env, check=True)
        finally:
            inp_queue.task_done()
    print(f"[GPU {gpu_index}] done.")


def main():
    # Reserve GPUs
    to_use = min(MAX_GPUS_TO_USE, len(list_gpus()))
    locks = find_and_lock_idle_gpus(to_use)
    gpu_indices = [idx for idx, _ in locks]
    print(f"[launcher] Reserved GPUs: {gpu_indices}")

    # Prepare work queue
    q = queue.Queue()
    for x in INPUTS:
        q.put(x)

    # Graceful unlock on Ctrl+C / exit
    def cleanup(*_):
        for _, lp in locks:
            unlock(lp)
        sys.exit(1)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Start one worker per GPU
    threads = []
    for idx in gpu_indices:
        t = threading.Thread(target=launch_worker, args=(idx, q), daemon=True)
        t.start()
        threads.append(t)

    # Wait for all work
    q.join()
    for t in threads:
        t.join()

    # Release locks
    for _, lp in locks:
        unlock(lp)
    print("[launcher] All jobs finished, locks released.")


if __name__ == "__main__":
    main()
