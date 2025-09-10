import csv
import numpy
import numpy as np
from pathlib import Path
from jormi.ww_io import io_manager, json_files

MODELS = ["linear", "quadratic"]


def load_best_fit_loglike(
    sim_dir: Path,
    model_name: str,
) -> float | None:
    data_path = io_manager.combine_file_path_parts(
        [sim_dir, model_name, "bin_per_t0", f"stage2_{model_name}_fitted_log_likelihoods.npy"],
    )
    data = np.load(data_path)
    return np.max(data)


def expected_model_from_mach(
    target_Mach: float,
) -> str:
    return "quadratic" if target_Mach > 1.0 else "linear"


def akaike_weight_linear(
    ll_lin: float,
    ll_quad: float,
) -> float:
    delta_aic = -2.0 * (ll_quad - ll_lin)
    if delta_aic > 1000: return 1.0
    elif delta_aic < -1000: return 0.0
    return 1.0 / (1.0 + numpy.exp(-0.5 * delta_aic))


def main():
    ## define paths
    script_dir = Path(__file__).parent
    dataset_dir = (script_dir / ".." / "datasets" / "backup").resolve()
    csv_path = dataset_dir / "model_comparison.csv"
    sim_dirs = io_manager.ItemFilter(
        include_string="Mach",
        include_files=False,
        include_folders=True,
    ).filter(directory=dataset_dir)
    csv_rows = []
    num_sims = 0
    agreement = 0
    for sim_dir in sim_dirs:
        sim_data_path = sim_dir / "dataset.json"
        sim_data = json_files.read_json_file_into_dict(sim_data_path, verbose=False)
        target_Mach = float(sim_data["plasma_params"]["target_Mach"])
        if target_Mach > 1: continue
        target_Re = float(sim_data["plasma_params"]["target_Re"])
        num_time_points = len(sim_data["measured_data"]["time_values"])
        ll_lin = load_best_fit_loglike(sim_dir, "linear")
        ll_quad = load_best_fit_loglike(sim_dir, "quadratic")
        linear_model_weight = akaike_weight_linear(ll_lin, ll_quad)
        best_model = "linear" if linear_model_weight >= 0.5 else "quadratic"
        expected_model = expected_model_from_mach(target_Mach)
        model_agreement = int(best_model == expected_model)
        num_sims += 1
        agreement += model_agreement
        csv_rows.append(
            {
                "sim_path": str(sim_dir),
                "target_Mach": target_Mach,
                "target_Re": target_Re,
                "num_time_points": num_time_points,
                "linear_model_weight": linear_model_weight,
                "best_model": best_model,
                "expected_model": expected_model,
                "model_agreement": model_agreement,
            },
        )
    csv_rows.sort(key=lambda row: (row["target_Mach"], row["target_Re"]))
    fieldnames = [
        "sim_path",
        "target_Mach",
        "target_Re",
        "num_time_points",
        "linear_model_weight",
        "best_model",
        "expected_model",
        "model_agreement",
    ]
    # csv_path.parent.mkdir(parents=True, exist_ok=True)
    # with open(csv_path, "w", newline="") as f:
    #     dw = csv.DictWriter(f, fieldnames=fieldnames)
    #     dw.writeheader()
    #     dw.writerows(csv_rows)
    print(f"Saved: {csv_path}")
    print(f"Compared: {num_sims}")
    print(f"Agreement: {agreement}/{num_sims} = {100 * agreement / num_sims:.1f}%")


if __name__ == "__main__":
    main()
