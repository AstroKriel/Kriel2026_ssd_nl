## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
import subprocess
import sys
from pathlib import Path

## personal
from jormi.ww_fns import parallel_dispatch
from jormi.ww_io import manage_io

##
## === GLOBAL PARAMS
##

ALLOW_OVERWRITE = False
MAX_PARALLEL_JOBS: int | None = 4

##
## === CONSTANTS
##

SCRIPT_DIR = Path(__file__).parent
UV_PROJECT = (SCRIPT_DIR / ".." / "..").resolve()
SIMS_DIR = (SCRIPT_DIR / ".." / ".." / "datasets" / "sims").resolve()


@dataclass(frozen=True)
class BinningConfig:
    tag: str
    num_bins: int | None


MODEL_TYPES = [
    "free",
    "linear",
    "quadratic",
]
BINNING_CONFIGS: list[BinningConfig] = [
    BinningConfig(
        tag="bin_per_t0",
        num_bins=None,
    ),
    BinningConfig(
        tag="100bins",
        num_bins=100,
    ),
]

##
## === HELPER FUNCTIONS
##


def output_exists(
    sim_directory: Path,
    model_name: str,
    binning_tag: str,
) -> bool:
    posterior_path = (
        sim_directory / model_name / binning_tag / f"stage2_{model_name}_fitted_posterior_samples.npy"
    )
    return posterior_path.exists()


def run_fit(
    sim_directory: Path,
    model_name: str,
    num_bins: int | None,
) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "fit_with_mcmc.py"),
        "--data-directory",
        str(sim_directory),
        "--model",
        model_name,
        "--no-progress",
    ]
    if num_bins is not None:
        cmd += ["--num-bins", str(num_bins)]
    subprocess.run(
        cmd,
        check=True,
        cwd=SCRIPT_DIR,
    )


##
## === MAIN PROGRAM
##


def main() -> None:
    all_sim_directories = manage_io.filter_directory(
        SIMS_DIR,
        req_include_words=["Mach", "Re", "Pm", "Nres"],
    )
    pending_jobs: list[tuple[Path, str, int | None]] = []
    num_existing_jobs = 0
    for sim_directory in sorted(all_sim_directories):
        for model_name in MODEL_TYPES:
            for binning_config in BINNING_CONFIGS:
                binning_tag = binning_config.tag
                num_bins = binning_config.num_bins
                if not ALLOW_OVERWRITE and output_exists(sim_directory, model_name, binning_tag):
                    num_existing_jobs += 1
                    continue
                pending_jobs.append((sim_directory, model_name, num_bins))
    if not pending_jobs:
        print("Nothing to fit.")
        return
    print(f"Found {len(pending_jobs)} missing fits; skipping {num_existing_jobs} existing fits.")
    parallel_dispatch.run_in_parallel(
        worker_fn=run_fit,
        grouped_args=pending_jobs,
        num_workers=MAX_PARALLEL_JOBS,
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
