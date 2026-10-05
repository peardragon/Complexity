"""Missing-input generation for the paper's Gaussian perceptron benchmark.

Dataset and hit-and-run recipes are recovered from the original pool scripts
in git b454f6a. Existing files are never replaced.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import truncnorm

STAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = STAGE_ROOT.parents[1]
DEFAULT_CONFIG = STAGE_ROOT / "config/default.json"
DEFAULT_GENERATION = {
    "dataset_seed_base": 20274541, "dataset_dimension_stride": 1000003,
    "dataset_index_stride": 9176, "reference_seed_base": 91337219,
    "reference_dimension_stride": 1000033, "reference_index_stride": 19937,
    "reference_burn": 5000, "reference_thin": 200, "margin_epsilon": 1e-10,
}


def generation_settings(config):
    values = {**DEFAULT_GENERATION, **config.get("input_generation", {})}
    if float(config["objective"]["lambda_ref"]) != 1.0:
        raise ValueError("the reference pool uses the paper's lambda_ref=1 Gaussian")
    if int(values["reference_burn"]) < 0 or int(values["reference_thin"]) < 1:
        raise ValueError("invalid hit-and-run burn/thin")
    if not 0 <= float(values["margin_epsilon"]) <= 1e-8:
        raise ValueError("invalid hard-feasibility numerical epsilon")
    return values


def dataset_path(project_root, n, dataset, output_root=None):
    root = Path(output_root) if output_root is not None else Path(project_root) / "01_theory/02_theory_sampling/raw_outputs/dataset_pool"
    return root / f"N_{n}" / f"dataset_{dataset+1:03d}" / "dataset.npz"


def reference_paths(project_root, n, dataset, count, output_root=None):
    root = Path(output_root) if output_root is not None else Path(project_root) / "01_theory/02_theory_sampling/raw_outputs/reference_pool"
    return [root / f"N_{n}" / f"dataset_{dataset+1:03d}" / f"ref_{ref+1:03d}" / "reference.npz" for ref in range(count)]


@contextmanager
def input_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".generation.lock").open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_npz(path, **payload):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def make_signed_gaussian_dataset(n, alpha, seed):
    rng = np.random.default_rng(int(seed))
    count = int(round(float(alpha)*n))
    if n <= 0 or not 0 < count < n:
        raise ValueError("invalid paper Gaussian-pattern dimensions")
    patterns = rng.normal(size=(count,n))
    labels = rng.choice([-1.0,1.0],size=count)
    return (labels[:,None]*patterns).astype(np.float64)


def ensure_dataset(config, n, dataset, *, project_root=PROJECT_ROOT, output_root=None):
    path = dataset_path(project_root,n,dataset,output_root)
    if path.is_file():
        return "skipped_existing"
    values = generation_settings(config)
    with input_lock(path.parent):
        if path.is_file():
            return "skipped_existing"
        seed = int(values["dataset_seed_base"])+int(values["dataset_dimension_stride"])*n+int(values["dataset_index_stride"])*dataset
        alpha = float(config["problem"]["alpha"])
        A = make_signed_gaussian_dataset(n,alpha,seed)
        atomic_npz(path,A=A,seed=np.asarray(seed),alpha=np.asarray(alpha))
    return "written"


def initial_feasible_reference(A):
    n = A.shape[1]
    theta, *_ = np.linalg.lstsq(A,np.full(A.shape[0],math.sqrt(n)),rcond=None)
    if float(np.min(A@theta/math.sqrt(n))) <= 1e-8:
        raise ValueError("failed to construct an interior hard-feasible reference")
    return theta.astype(np.float64)


def sample_truncated_normal(mean, lower, upper, rng):
    lo = float(ndtr(lower-mean)) if np.isfinite(lower) else 0.0
    hi = float(ndtr(upper-mean)) if np.isfinite(upper) else 1.0
    lo,hi = float(np.clip(lo,0,1)),float(np.clip(hi,0,1))
    if hi <= lo:
        # Stable conditional-Gaussian tail, not the old point-mass clipping.
        return float(truncnorm.rvs(lower-mean,upper-mean,loc=mean,random_state=rng))
    u = float(np.clip(rng.uniform(lo,hi),np.nextafter(0.,1.),np.nextafter(1.,0.)))
    return float(mean+ndtri(u))


def hit_and_run_references(A, *, count, burn, thin, seed, margin_epsilon):
    rng = np.random.default_rng(int(seed))
    theta = initial_feasible_reference(A)
    n = A.shape[1]
    required_margin = float(margin_epsilon)*math.sqrt(n)
    references = []
    for step in range(int(burn)+int(count)*int(thin)):
        direction = rng.normal(size=n)
        direction /= np.linalg.norm(direction)
        current,slope = A@theta,A@direction
        positive,negative = slope>1e-14,slope < -1e-14
        lower = float(np.max((required_margin-current[positive])/slope[positive])) if np.any(positive) else -np.inf
        upper = float(np.min((required_margin-current[negative])/slope[negative])) if np.any(negative) else np.inf
        if lower >= upper:
            raise ValueError("no feasible hit-and-run interval")
        step_size = sample_truncated_normal(float(-theta@direction),lower,upper,rng)
        theta = theta+step_size*direction
        if step >= int(burn) and (step-int(burn))%int(thin)==0:
            references.append(theta.copy())
    output = np.asarray(references[:count],dtype=np.float64)
    if output.shape != (count,n) or not np.isfinite(output).all() or np.min(output@A.T) <= 0:
        raise ValueError("invalid generated hard-reference pool")
    return output


def ensure_references(config, n, dataset, *, project_root=PROJECT_ROOT, dataset_root=None, output_root=None):
    count = int(config["problem"]["references_per_dataset"])
    paths = reference_paths(project_root,n,dataset,count,output_root)
    missing = [path for path in paths if not path.is_file()]
    if not missing:
        return {"written":0,"skipped_existing":count}
    values = generation_settings(config)
    ensure_dataset(config,n,dataset,project_root=project_root,output_root=dataset_root)
    with input_lock(paths[0].parent.parent):
        if all(path.is_file() for path in paths):
            return {"written":0,"skipped_existing":count}
        source = dataset_path(project_root,n,dataset,dataset_root)
        with np.load(source,allow_pickle=False) as payload:
            A = np.asarray(payload["A"],dtype=np.float64)
        expected = (int(round(float(config["problem"]["alpha"])*n)),n)
        if A.shape != expected or not np.isfinite(A).all():
            raise ValueError(f"invalid Gaussian input: {source}")
        seed = int(values["reference_seed_base"])+int(values["reference_dimension_stride"])*n+int(values["reference_index_stride"])*dataset
        refs = hit_and_run_references(A,count=count,burn=int(values["reference_burn"]),thin=int(values["reference_thin"]),seed=seed,margin_epsilon=float(values["margin_epsilon"]))
        written = 0
        for path,theta in zip(paths,refs,strict=True):
            if path.is_file():
                continue
            path.parent.mkdir(parents=True,exist_ok=True)
            atomic_npz(path,theta=theta);written += 1
        return {"written":written,"skipped_existing":count-written}


def ensure_row_inputs(config, rows, project_root=PROJECT_ROOT):
    coordinates = sorted({(int(row["N"]),int(row["dataset_id"])) for row in rows})
    for n,dataset in coordinates:
        ensure_dataset(config,n,dataset,project_root=project_root)
        ensure_references(config,n,dataset,project_root=project_root)


def main(kind):
    parser = argparse.ArgumentParser(description=f"Reuse or generate the paper theory {kind} pool.")
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    parser.add_argument("--output-root",type=Path)
    parser.add_argument("--dataset-root",type=Path)
    parser.add_argument("--n-values",help="Comma-separated subset of the configured dimensions.")
    parser.add_argument("--dataset-index",type=int,help="Select one zero-based dataset index.")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--check-only",action="store_true")
    args = parser.parse_args()
    if args.check_only and args.execute:parser.error("--check-only is read-only")
    config = json.loads(args.config.read_text())
    generation_settings(config)
    configured = [int(n) for n in config["problem"]["dimensions"]]
    dimensions = [int(n) for n in args.n_values.split(",")] if args.n_values else configured
    datasets = [args.dataset_index] if args.dataset_index is not None else list(range(int(config["problem"]["datasets_per_dimension"])))
    if not dimensions or len(set(dimensions))!=len(dimensions) or any(n not in configured for n in dimensions):parser.error("dimensions must be a unique configured subset")
    if any(not 0<=d<int(config["problem"]["datasets_per_dimension"]) for d in datasets):parser.error("dataset index outside configured pool")
    report = {"written":0,"skipped_existing":0,"missing":0}
    for n in dimensions:
        for dataset in datasets:
            paths = ([dataset_path(PROJECT_ROOT,n,dataset,args.output_root)] if kind=="dataset"
                     else reference_paths(PROJECT_ROOT,n,dataset,int(config["problem"]["references_per_dataset"]),args.output_root))
            missing = [path for path in paths if not path.is_file()]
            if not missing:
                report["skipped_existing"] += len(paths);continue
            if args.check_only:
                raise FileNotFoundError(missing[0])
            if not args.execute:
                report["missing"] += len(missing);continue
            if kind=="dataset":
                report[ensure_dataset(config,n,dataset,output_root=args.output_root)] += 1
            else:
                result = ensure_references(config,n,dataset,dataset_root=args.dataset_root,output_root=args.output_root)
                for key,value in result.items():report[key] += value
    print(json.dumps({"status":"complete" if report["missing"]==0 else "planned",**report},indent=2))
