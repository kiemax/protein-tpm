import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from proteintpm.tp_mapping.jobs import BoxSource, ResidueSelector
from proteintpm.tp_mapping.predict import run_mapping_job
from proteintpm.tp_mapping.setup import setup_mapping_job


@dataclass(frozen=True)
class BatchJobRow:
    """
    One tp_mapping batch job specification.

    Supported box modes:
      - ligand  : box defined by ligand coordinates
      - chains  : box defined by all atoms in target_chains
      - residue : box defined by one residue within target_chains
    """

    pdb: str
    target_chains: list[str]
    map_name: str
    model_logs: str

    ligand: str | None = None
    box_mode: str = "ligand"
    box_chain: str | None = None
    box_res_num: str | None = None
    box_res_name: str | None = None

    voxel_size: float | None = None
    padding: float | None = None
    batch_size: int | None = None


# ----------------------------
# CSV parsing
# ----------------------------


def _optional_str(row: dict, key: str) -> str | None:
    value = row.get(key, "")
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def _optional_float(row: dict, key: str) -> float | None:
    value = _optional_str(row, key)
    return float(value) if value is not None else None


def _optional_int(row: dict, key: str) -> int | None:
    value = _optional_str(row, key)
    return int(value) if value is not None else None


def parse_jobs_csv(csv_path: Path) -> list[BatchJobRow]:
    """
    Parse a batch jobs CSV.

    Required columns:
      pdb,target_chains,map_name,model_logs

    Optional columns:
      ligand,box_mode,box_chain,box_res_num,box_res_name,
      voxel_size,padding,batch_size

    Notes:
      - target_chains is whitespace-separated, e.g. "A B"
      - ligand is required when box_mode=ligand
      - box_chain + box_res_num are required when box_mode=residue
    """
    jobs: list[BatchJobRow] = []

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(
            line for line in f if line.strip() and not line.lstrip().startswith("#")
        )

        required = {"pdb", "target_chains", "map_name", "model_logs"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for line_no, row in enumerate(reader, start=2):
            chains_raw = _optional_str(row, "target_chains")
            if chains_raw is None:
                raise ValueError(f"Empty target_chains at line {line_no}")

            target_chains = chains_raw.split()
            if not target_chains:
                raise ValueError(f"Empty target_chains at line {line_no}")

            box_mode = (_optional_str(row, "box_mode") or "ligand").lower()

            ligand = _optional_str(row, "ligand")
            box_chain = _optional_str(row, "box_chain")
            box_res_num = _optional_str(row, "box_res_num")
            box_res_name = _optional_str(row, "box_res_name")

            if box_mode not in {"ligand", "chains", "residue"}:
                raise ValueError(
                    f"Invalid box_mode at line {line_no}: {box_mode!r}. "
                    "Use one of: ligand, chains, residue."
                )

            if box_mode == "ligand" and ligand is None:
                raise ValueError(
                    f"box_mode=ligand requires 'ligand' column value at line {line_no}"
                )

            if box_mode == "residue":
                if box_chain is None or box_res_num is None:
                    raise ValueError(
                        f"box_mode=residue requires box_chain and box_res_num at line {line_no}"
                    )
                # if box_chain not in target_chains:
                #    raise ValueError(
                #        f"box_chain={box_chain!r} must be included in target_chains "
                #        f"at line {line_no}"
                #    )

            pdb = _optional_str(row, "pdb")
            map_name = _optional_str(row, "map_name")
            model_logs = _optional_str(row, "model_logs")

            if pdb is None:
                raise ValueError(f"Empty pdb at line {line_no}")
            if map_name is None:
                raise ValueError(f"Empty map_name at line {line_no}")
            if model_logs is None:
                raise ValueError(f"Empty model_logs at line {line_no}")

            jobs.append(
                BatchJobRow(
                    pdb=pdb,
                    target_chains=target_chains,
                    map_name=map_name,
                    model_logs=model_logs,
                    ligand=ligand,
                    box_mode=box_mode,
                    box_chain=box_chain,
                    box_res_num=box_res_num,
                    box_res_name=box_res_name,
                    voxel_size=_optional_float(row, "voxel_size"),
                    padding=_optional_float(row, "padding"),
                    batch_size=_optional_int(row, "batch_size"),
                )
            )

    if not jobs:
        raise ValueError(f"No jobs found in CSV: {csv_path}")

    return jobs


# ----------------------------
# Job-row conversion
# ----------------------------


def build_box_source(job: BatchJobRow) -> BoxSource:
    mode = job.box_mode.strip().lower()

    if mode == "ligand":
        return BoxSource(mode="ligand")

    if mode == "chains":
        return BoxSource(mode="chains")

    if mode == "residue":
        if job.box_chain is None or job.box_res_num is None:
            raise ValueError("Residue box mode requires box_chain and box_res_num.")
        return BoxSource(
            mode="residue",
            residue=ResidueSelector(
                chain_id=job.box_chain,
                res_num=job.box_res_num,
                res_name=job.box_res_name,
            ),
        )

    raise RuntimeError(f"Unhandled box mode: {mode}")


# ----------------------------
# Single-job workers
# ----------------------------

def _run_single_job(job_dir: Path, device: str, overwrite: bool, batch_size: int) -> None:
    """Module-level worker: runs exactly one inference job, then the process
    that ran it is torn down by the pool (max_tasks_per_child=1), so ALL of its
    memory -- including the TF/Keras graph from load_model -- is reclaimed by
    the OS. This is the reliable fix for the load_model-every-job leak that
    clear_session() does not fully release."""
    infer_one(
        job_dir=job_dir,
        device=device,
        overwrite=overwrite,
        batch_size=batch_size,
    )



def setup_one(
    job: BatchJobRow,
    outdir: Path,
    default_voxel_size: float,
    default_padding: float,
    run_id: str | None,
    model_link_mode: str,
    overwrite: bool,
) -> Path:
    voxel_size = job.voxel_size if job.voxel_size is not None else default_voxel_size
    padding = job.padding if job.padding is not None else default_padding

    box_source = build_box_source(job)

    mapping_job = setup_mapping_job(
        pdb_path=Path(job.pdb),
        target_chains=job.target_chains,
        ligand_path=None if job.ligand is None else Path(job.ligand),
        box_source=box_source,
        map_name=job.map_name,
        model_logs=Path(job.model_logs),
        outdir=outdir,
        voxel_size=voxel_size,
        padding=padding,
        run_id=run_id,
        model_link_mode=model_link_mode,
        overwrite=overwrite,
    )

    return Path(mapping_job.outputs.output_dir)


def infer_one(
    job_dir: Path,
    device: str,
    overwrite: bool,
    batch_size: int,
) -> dict:
    return run_mapping_job(
        job=str(job_dir),
        device=device,
        overwrite=overwrite,
        batch_size=batch_size,
    )


#def gpu_worker(
#    gpu_index: int,
#    jobs_with_batch_size: list[tuple[Path, int]],
#    overwrite: bool,
#) -> tuple[int, int]:
#    """
#    Run multiple job_dirs sequentially on a single GPU.
#    Returns (gpu_index, num_done).
#    """
#    done = 0
#    device = f"gpu:{gpu_index}"
#
#    for job_dir, batch_size in jobs_with_batch_size:
#        infer_one(
#            job_dir=job_dir,
#            device=device,
#            overwrite=overwrite,
#            batch_size=batch_size,
#        )
#        done += 1
#
#    return gpu_index, done
#
def gpu_worker(
    gpu_index: int,
    jobs_with_batch_size: list[tuple[Path, int]],
    overwrite: bool,
) -> tuple[int, int]:
    """
    Run job_dirs on a single GPU, but each in its OWN subprocess so memory is
    fully released between jobs (TF leaks across load_model calls in a
    persistent process; a fresh process per job sidesteps that entirely).
    Returns (gpu_index, num_done).
    """
    device = f"gpu:{gpu_index}"
    done = 0

    # one worker, recycled after every single job
    with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as executor:
        futures = [
            executor.submit(_run_single_job, job_dir, device, overwrite, batch_size)
            for job_dir, batch_size in jobs_with_batch_size
        ]
        for fut in futures:
            fut.result()   # re-raise any job error, same as before
            done += 1

    return gpu_index, done



def chunk_round_robin(items: list[tuple[Path, int]], n: int) -> list[list[tuple[Path, int]]]:
    if n <= 0:
        raise ValueError("n must be > 0.")

    buckets: list[list[tuple[Path, int]]] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        buckets[i % n].append(item)
    return buckets


# ----------------------------
# Batch entrypoint
# ----------------------------


def run_mapping_batch(
    *,
    jobs_csv: Path,
    outdir: Path,
    voxel_size: float = 0.5,
    padding: float = 1.0,
    run_id: str | None = None,
    model_link_mode: str = "symlink",
    setup_workers: int = 4,
    mode: str = "gpu",
    gpus: str = "0",
    infer_workers: int = 2,
    batch_size: int = 0,
    overwrite: bool = False,
) -> list[Path]:
    """
    Run batch setup + prediction for tp_mapping jobs.

    batch_size acts as the global default. A per-row batch_size in the CSV
    overrides it for that job.

    Returns:
        Sorted list of created job directories.
    """
    mode = mode.strip().lower()
    if mode not in {"gpu", "cpu"}:
        raise ValueError("mode must be one of: gpu, cpu")

    if model_link_mode not in {"symlink", "copy", "path_only"}:
        raise ValueError("model_link_mode must be one of: symlink, copy, path_only")

    jobs = parse_jobs_csv(jobs_csv)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[tp_mapping.batch] running setup for {len(jobs)} jobs with {setup_workers} workers")

    indexed_jobs = list(enumerate(jobs))
    setup_results: list[tuple[int, Path]] = []

    with ProcessPoolExecutor(max_workers=setup_workers) as executor:
        futures = [
            executor.submit(
                setup_one,
                job=job,
                outdir=outdir,
                default_voxel_size=voxel_size,
                default_padding=padding,
                run_id=run_id,
                model_link_mode=model_link_mode,
                overwrite=overwrite,
            )
            for _, job in indexed_jobs
        ]

        for (idx, _), future in zip(indexed_jobs, futures):
            setup_results.append((idx, future.result()))

    setup_results.sort(key=lambda x: x[0])
    job_dirs = [job_dir for _, job_dir in setup_results]

    jobs_with_batch_size: list[tuple[Path, int]] = []
    for job, job_dir in zip(jobs, job_dirs):
        effective_batch_size = job.batch_size if job.batch_size is not None else batch_size
        jobs_with_batch_size.append((job_dir, effective_batch_size))

    if mode == "cpu":
        print(
            f"[tp_mapping.batch] running CPU prediction for {len(job_dirs)} jobs "
            f"with {infer_workers} workers"
        )

        with ProcessPoolExecutor(max_workers=infer_workers) as executor:
            futures = [
                executor.submit(
                    infer_one,
                    job_dir=job_dir,
                    device="cpu",
                    overwrite=overwrite,
                    batch_size=effective_batch_size,
                )
                for job_dir, effective_batch_size in jobs_with_batch_size
            ]

            for future in as_completed(futures):
                future.result()

    else:
        gpu_list = [int(x.strip()) for x in gpus.split(",") if x.strip()]
        if not gpu_list:
            raise ValueError("No GPUs specified. Use gpus='0' or gpus='0,1'.")

        print(
            f"[tp_mapping.batch] running GPU prediction for {len(job_dirs)} jobs on GPUs {gpu_list}"
        )

        buckets = chunk_round_robin(jobs_with_batch_size, len(gpu_list))

        with ProcessPoolExecutor(max_workers=len(gpu_list)) as executor:
            futures = [
                executor.submit(
                    gpu_worker,
                    gpu_index=gpu_idx,
                    jobs_with_batch_size=bucket,
                    overwrite=overwrite,
                )
                for gpu_idx, bucket in zip(gpu_list, buckets)
            ]

            for future in as_completed(futures):
                gpu_idx, done = future.result()
                print(f"[tp_mapping.batch] GPU {gpu_idx} finished {done} jobs")

    print("[tp_mapping.batch] DONE")
    return job_dirs
