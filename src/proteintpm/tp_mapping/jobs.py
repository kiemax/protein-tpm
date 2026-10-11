import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ----------------------------
# Small helpers
# ----------------------------


def make_model_id(model_path: Path) -> str:
    """
    Stable, human-friendly model identifier:
    <basename>__<sha1-of-absolute-path-8chars>
    """
    p = model_path.resolve()
    base = p.name
    h = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:8]
    return f"{base}__{h}"


def make_run_id(explicit: str | None = None) -> str:
    """
    Collision-resistant run id. Uses local time by default unless explicit is given.
    """
    if explicit:
        return explicit
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def utc_now_iso() -> str:
    """
    Return current UTC timestamp in ISO format with trailing Z.
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def resolve_job_dir(job: str | None, mapdata_path: str | None) -> Path:
    """
    Determine the directory holding job.json (new layout) or setup files (legacy layout).
    Kept here because both setup/predict CLIs may need the same resolution logic.
    """
    if job and mapdata_path:
        raise ValueError("Use either --job or --mapdata_path, not both.")

    if job:
        job_dir = Path(job)
        if job_dir.is_file() and job_dir.name == "job.json":
            job_dir = job_dir.parent
        if not (job_dir / "job.json").exists():
            raise FileNotFoundError(f"Could not find job.json in: {job_dir}")
        return job_dir

    if mapdata_path:
        return Path(mapdata_path)

    raise ValueError("You must provide --job (preferred) or --mapdata_path (legacy).")


# ----------------------------
# Selection model
# ----------------------------


@dataclass
class ResidueSelector:
    """
    Optional residue-level restriction for map setup.

    res_num is stored as string to match proteintpm.data.pdb_io.create_dataframe(),
    which keeps residue numbers as stripped strings.
    insertion_code support is implicit: users can pass values like '42A' in res_num.
    """

    chain_id: str
    res_num: str
    res_name: str | None = None

    def normalized(self) -> "ResidueSelector":
        chain_id = self.chain_id.strip()
        res_num = str(self.res_num).strip()
        res_name = self.res_name.strip().upper() if self.res_name is not None else None

        if not chain_id:
            raise ValueError("ResidueSelector.chain_id must not be empty.")
        if not res_num:
            raise ValueError("ResidueSelector.res_num must not be empty.")

        return ResidueSelector(
            chain_id=chain_id,
            res_num=res_num,
            res_name=res_name,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResidueSelector":
        return cls(**payload).normalized()


@dataclass
class BoxSource:
    """
    Defines how the map box / voxel grid bounding region is chosen.

    mode:
      - 'ligand': use ligand coordinates
      - 'chains': use all atoms from selected target chains
      - 'residue': use atoms from a specific residue
    """

    mode: str = "ligand"
    residue: ResidueSelector | None = None

    def normalized(self) -> "BoxSource":
        mode = self.mode.strip().lower()

        if mode not in {"ligand", "chains", "residue"}:
            raise ValueError(
                f"Unsupported box source mode: {self.mode!r}. Use one of: ligand, chains, residue."
            )

        residue = self.residue
        if mode == "residue":
            if residue is None:
                raise ValueError("BoxSource.mode='residue' requires a residue selector.")
            residue = residue.normalized()
        else:
            if residue is not None:
                raise ValueError(f"BoxSource.mode='{mode}' must not define residue selection.")

        return BoxSource(mode=mode, residue=residue)

    def to_dict(self) -> dict[str, Any]:
        payload = {"mode": self.mode}
        if self.residue is not None:
            payload["residue"] = self.residue.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BoxSource":
        residue_payload = payload.get("residue")
        residue = (
            ResidueSelector.from_dict(residue_payload) if residue_payload is not None else None
        )
        return cls(
            mode=payload.get("mode", "ligand"),
            residue=residue,
        ).normalized()


# ----------------------------
# Job model
# ----------------------------


@dataclass
class MappingInputs:
    pdb: str
    ligand: str | None
    target_chains: list[str]
    box_source: BoxSource = field(default_factory=BoxSource)

    def normalized(self) -> "MappingInputs":
        pdb = str(self.pdb).strip()
        ligand = None if self.ligand is None else str(self.ligand).strip()

        target_chains = sorted({c.strip() for c in self.target_chains if c.strip()})
        if not pdb:
            raise ValueError("inputs.pdb must not be empty.")
        if not target_chains:
            raise ValueError("inputs.target_chains must contain at least one chain.")

        box_source = self.box_source.normalized()

        if box_source.mode == "ligand" and not ligand:
            raise ValueError("inputs.ligand is required when box_source.mode='ligand'.")

        # if box_source.mode == "residue":
        #    assert box_source.residue is not None
        #    if box_source.residue.chain_id not in target_chains:
        #        raise ValueError(
        #            "ResidueSelector.chain_id must be included in inputs.target_chains."
        #        )

        return MappingInputs(
            pdb=pdb,
            ligand=ligand,
            target_chains=target_chains,
            box_source=box_source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pdb": self.pdb,
            "ligand": self.ligand,
            "target_chains": self.target_chains,
            "box_source": self.box_source.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MappingInputs":
        return cls(
            pdb=payload["pdb"],
            ligand=payload.get("ligand"),
            target_chains=list(payload["target_chains"]),
            box_source=BoxSource.from_dict(payload.get("box_source", {"mode": "ligand"})),
        ).normalized()


@dataclass
class GridSpec:
    voxel_size: float
    padding: float

    def normalized(self) -> "GridSpec":
        voxel_size = float(self.voxel_size)
        padding = float(self.padding)

        if voxel_size <= 0:
            raise ValueError("grid.voxel_size must be > 0.")
        if padding < 0:
            raise ValueError("grid.padding must be >= 0.")

        return GridSpec(voxel_size=voxel_size, padding=padding)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GridSpec":
        return cls(**payload).normalized()


@dataclass
class ModelMaterialized:
    mode: str
    path: str
    ok: bool

    def normalized(self) -> "ModelMaterialized":
        mode = self.mode.strip().lower()
        if mode not in {"copy", "symlink", "path_only"}:
            raise ValueError(f"Unsupported materialization mode: {self.mode!r}.")
        return ModelMaterialized(
            mode=mode,
            path=str(self.path).strip(),
            ok=bool(self.ok),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelMaterialized":
        return cls(**payload).normalized()


@dataclass
class ModelSpec:
    model_logs: str
    model_id: str
    materialized: ModelMaterialized

    def normalized(self) -> "ModelSpec":
        model_logs = str(self.model_logs).strip()
        model_id = str(self.model_id).strip()

        if not model_logs:
            raise ValueError("model.model_logs must not be empty.")
        if not model_id:
            raise ValueError("model.model_id must not be empty.")

        return ModelSpec(
            model_logs=model_logs,
            model_id=model_id,
            materialized=self.materialized.normalized(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_logs": self.model_logs,
            "model_id": self.model_id,
            "materialized": self.materialized.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelSpec":
        return cls(
            model_logs=payload["model_logs"],
            model_id=payload["model_id"],
            materialized=ModelMaterialized.from_dict(payload["materialized"]),
        ).normalized()


@dataclass
class OutputSpec:
    outdir: str
    output_dir: str
    setup_dir: str
    inference_dir: str
    model_dir: str
    map_name: str

    def normalized(self) -> "OutputSpec":
        map_name = str(self.map_name).strip()
        if not map_name:
            raise ValueError("outputs.map_name must not be empty.")

        return OutputSpec(
            outdir=str(self.outdir).strip(),
            output_dir=str(self.output_dir).strip(),
            setup_dir=str(self.setup_dir).strip(),
            inference_dir=str(self.inference_dir).strip(),
            model_dir=str(self.model_dir).strip(),
            map_name=map_name,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OutputSpec":
        return cls(**payload).normalized()


@dataclass
class ArtifactSpec:
    atom_cloud: str
    voxel_centers: str
    grid_x: str
    grid_y: str
    grid_z: str
    data_target: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ArtifactSpec":
        return cls(**payload)


@dataclass
class MappingJob:
    created_at: str
    run_id: str
    inputs: MappingInputs
    grid: GridSpec
    model: ModelSpec
    outputs: OutputSpec
    artifacts: ArtifactSpec

    def normalized(self) -> "MappingJob":
        created_at = str(self.created_at).strip()
        run_id = str(self.run_id).strip()

        if not created_at:
            raise ValueError("created_at must not be empty.")
        if not run_id:
            raise ValueError("run_id must not be empty.")

        return MappingJob(
            created_at=created_at,
            run_id=run_id,
            inputs=self.inputs.normalized(),
            grid=self.grid.normalized(),
            model=self.model.normalized(),
            outputs=self.outputs.normalized(),
            artifacts=self.artifacts,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "run_id": self.run_id,
            "inputs": self.inputs.to_dict(),
            "grid": self.grid.to_dict(),
            "model": self.model.to_dict(),
            "outputs": self.outputs.to_dict(),
            "artifacts": self.artifacts.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MappingJob":
        return cls(
            created_at=payload["created_at"],
            run_id=payload["run_id"],
            inputs=MappingInputs.from_dict(payload["inputs"]),
            grid=GridSpec.from_dict(payload["grid"]),
            model=ModelSpec.from_dict(payload["model"]),
            outputs=OutputSpec.from_dict(payload["outputs"]),
            artifacts=ArtifactSpec.from_dict(payload["artifacts"]),
        ).normalized()

    @classmethod
    def from_json(cls, path: Path) -> "MappingJob":
        return cls.from_dict(load_json(path))

    def write(self, path: Path) -> None:
        write_json(path, self.to_dict())


# ----------------------------
# Naming helpers
# ----------------------------


def chain_label(target_chains: list[str]) -> str:
    chains = sorted({c.strip() for c in target_chains if c.strip()})
    if not chains:
        raise ValueError("target_chains must contain at least one non-empty chain.")
    return "+".join(chains)


def residue_label(residue: ResidueSelector) -> str:
    residue = residue.normalized()
    label = f"{residue.chain_id}_{residue.res_num}"
    if residue.res_name is not None:
        label = f"{label}_{residue.res_name}"
    return label


def box_label(box_source: BoxSource) -> str:
    box_source = box_source.normalized()

    if box_source.mode == "ligand":
        return "box_ligand"
    if box_source.mode == "chains":
        return "box_chains"
    if box_source.mode == "residue":
        assert box_source.residue is not None
        return f"box_residue_{residue_label(box_source.residue)}"

    raise RuntimeError(f"Unhandled box source mode: {box_source.mode}")


def make_output_dir(
    outdir: Path,
    pdb_path: Path,
    target_chains: list[str],
    model_id: str,
    map_name: str,
    run_id: str,
    ligand_path: Path | None = None,
    box_source: BoxSource | None = None,
) -> Path:
    """
    Build the canonical output directory path.

    Examples:
      ligand box:
        <outdir>/<pdb_stem>/chain_A+B/ligand_<ligand_stem>/model_<model_id>/map_<map_name>__<run_id>

      chain box:
        <outdir>/<pdb_stem>/chain_A+B/box_chains/model_<model_id>/map_<map_name>__<run_id>

      residue box:
        <outdir>/<pdb_stem>/chain_A+B/box_residue_A_42_TYR/model_<model_id>/map_<map_name>__<run_id>
    """
    box_source = box_source.normalized() if box_source is not None else BoxSource(mode="ligand")

    parts = [
        outdir,
        pdb_path.stem,
        f"chain_{chain_label(target_chains)}",
    ]

    if box_source.mode == "ligand":
        if ligand_path is None:
            raise ValueError("ligand_path is required when box_source.mode='ligand'.")
        parts.append(f"ligand_{ligand_path.stem}")
    else:
        parts.append(box_label(box_source))

    parts.extend(
        [
            f"model_{model_id}",
            f"map_{map_name}__{run_id}",
        ]
    )

    return Path(*parts)


def make_default_artifacts() -> ArtifactSpec:
    return ArtifactSpec(
        atom_cloud="setup/atom_cloud.npy",
        voxel_centers="setup/voxel_centers.npy",
        grid_x="setup/grid_x.npy",
        grid_y="setup/grid_y.npy",
        grid_z="setup/grid_z.npy",
        data_target="setup/data_target.pkl",
    )


def build_mapping_job(
    *,
    pdb_path: Path,
    target_chains: list[str],
    ligand_path: Path | None,
    box_source: BoxSource,
    voxel_size: float,
    padding: float,
    model_logs: Path,
    model_materialized: ModelMaterialized,
    outdir: Path,
    output_dir: Path,
    setup_dir: Path,
    inference_dir: Path,
    model_dir: Path,
    map_name: str,
    run_id: str,
) -> MappingJob:
    model_id = make_model_id(model_logs)

    return MappingJob(
        created_at=utc_now_iso(),
        run_id=run_id,
        inputs=MappingInputs(
            pdb=str(pdb_path.resolve()),
            ligand=None if ligand_path is None else str(ligand_path.resolve()),
            target_chains=target_chains,
            box_source=box_source,
        ),
        grid=GridSpec(
            voxel_size=voxel_size,
            padding=padding,
        ),
        model=ModelSpec(
            model_logs=str(model_logs.resolve()),
            model_id=model_id,
            materialized=model_materialized,
        ),
        outputs=OutputSpec(
            outdir=str(outdir.resolve()),
            output_dir=str(output_dir.resolve()),
            setup_dir=str(setup_dir.resolve()),
            inference_dir=str(inference_dir.resolve()),
            model_dir=str(model_dir.resolve()),
            map_name=map_name,
        ),
        artifacts=make_default_artifacts(),
    ).normalized()
