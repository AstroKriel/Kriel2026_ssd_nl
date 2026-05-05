## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import math
from pathlib import Path
from typing import TypeAlias, cast

## personal
from jormi.ww_io import json_io

##
## === TYPE ALIASES
##

StatsBlock: TypeAlias = dict[str, float]
SuiteStats: TypeAlias = dict[str, dict[str, float | StatsBlock]]
SuiteDataset: TypeAlias = dict[str, SuiteStats]


##
## === HELPER FUNCTIONS
##

def compute_scientific_basis(
    value: float,
) -> tuple[float, int]:
    """Return (mantissa, exponent) so that value = mantissa * 10**exponent."""
    if not math.isfinite(value) or value == 0:
        return 0.0, 0
    exponent = int(
        math.floor(
            math.log10(
                abs(
                    value,
                ),
            ),
        ),
    )
    mantissa = value / (10**exponent)
    if abs(mantissa) < 1:
        exponent -= 1
        mantissa = value / (10**exponent)
    return mantissa, exponent


def compute_decimals_from_max_error(
    error: float,
) -> int:
    """Use one significant digit for the largest uncertainty."""
    if not math.isfinite(error) or error <= 0:
        return 0
    exponent = math.floor(
        math.log10(
            error,
        ),
    )
    return max(0, -int(exponent))


def create_fixed_decimal_string(
    *,
    value: float,
    decimals: int,
) -> str:
    """Format with a fixed number of decimals (no scientific notation)."""
    format_string = f"{{:.{decimals}f}}"
    return format_string.format(value)


def create_scientific_value_string(
    *,
    value: float,
    num_significant_digits: int = 2,
) -> str:
    """Format a plain value (no errors) in scientific notation for LaTeX math."""
    if value == 0 or not math.isfinite(value):
        return "$0$"
    exponent = int(
        math.floor(
            math.log10(
                abs(
                    value,
                ),
            ),
        ),
    )
    mantissa = value / (10**exponent)
    mantissa_str = f"{mantissa:.{num_significant_digits}g}"
    if exponent == 0:
        return f"${mantissa_str}$"
    return f"${mantissa_str}\\times 10^{{{exponent}}}$"


def compute_linear_errors_from_log10_stats(
    *,
    median_log10: float,
    error_log10_lo: float,
    error_log10_hi: float,
) -> tuple[float, float, float]:
    """Convert log10 median and asymmetric deltas to the linear domain."""
    median_value = 10.0**median_log10
    lower_value = 10.0**(median_log10 - error_log10_lo)
    upper_value = 10.0**(median_log10 + error_log10_hi)
    return median_value, (median_value - lower_value), (upper_value - median_value)


def create_nonzero_uncertainty_string(
    *,
    uncertainty_str: str,
    decimals: int,
) -> str:
    """Ensure an uncertainty string never rounds to zero."""
    if float(uncertainty_str) == 0.0:
        return f"{0.1:.{decimals}f}"
    return uncertainty_str


def create_scientific_errorbar_string(
    *,
    value: float,
    error_lo: float,
    error_hi: float,
) -> str:
    """Format a value and asymmetric errors in a compact scientific basis."""
    mantissa, exponent = compute_scientific_basis(value)
    scale = 10**exponent
    scaled_error_lo = error_lo / scale
    scaled_error_hi = error_hi / scale
    max_error = max(scaled_error_lo, scaled_error_hi)
    decimals = min(
        compute_decimals_from_max_error(max_error),
        2,
    )
    mantissa_str = create_fixed_decimal_string(
        value=mantissa,
        decimals=decimals,
    )
    error_lo_str = create_fixed_decimal_string(
        value=scaled_error_lo,
        decimals=decimals,
    )
    error_hi_str = create_fixed_decimal_string(
        value=scaled_error_hi,
        decimals=decimals,
    )
    error_lo_str = create_nonzero_uncertainty_string(
        uncertainty_str=error_lo_str,
        decimals=decimals,
    )
    error_hi_str = create_nonzero_uncertainty_string(
        uncertainty_str=error_hi_str,
        decimals=decimals,
    )
    if exponent == 0:
        return f"${mantissa_str}_{{-{error_lo_str}}}^{{+{error_hi_str}}}$"
    errorbar_core = f"\\left({mantissa_str}_{{-{error_lo_str}}}^{{+{error_hi_str}}}\\right)"
    return f"${errorbar_core}\\times 10^{{{exponent}}}$"


def create_log10_stats_string(
    stats_block: StatsBlock,
) -> str:
    value, error_lo, error_hi = compute_linear_errors_from_log10_stats(
        median_log10=stats_block["p50"],
        error_log10_lo=stats_block["std_lo"],
        error_log10_hi=stats_block["std_hi"],
    )
    return create_scientific_errorbar_string(
        value=value,
        error_lo=error_lo,
        error_hi=error_hi,
    )


def create_linear_stats_string(
    stats_block: StatsBlock,
) -> str:
    return create_scientific_errorbar_string(
        value=stats_block["p50"],
        error_lo=stats_block["std_lo"],
        error_hi=stats_block["std_hi"],
    )


def create_duration_stats_string(
    stats_block: StatsBlock,
) -> str:
    """Format nonlinear duration with one decimal place and no scientific basis."""
    value, error_lo, error_hi = compute_linear_errors_from_log10_stats(
        median_log10=stats_block["p50"],
        error_log10_lo=stats_block["std_lo"],
        error_log10_hi=stats_block["std_hi"],
    )
    value_str = f"{value:.1f}"
    error_lo_str = f"{error_lo:.1f}"
    error_hi_str = f"{error_hi:.1f}"
    error_lo_str = create_nonzero_uncertainty_string(
        uncertainty_str=error_lo_str,
        decimals=1,
    )
    error_hi_str = create_nonzero_uncertainty_string(
        uncertainty_str=error_hi_str,
        decimals=1,
    )
    return f"${value_str}_{{-{error_lo_str}}}^{{+{error_hi_str}}}$"


def create_simulations_at_resolution_string(
    *,
    count: int,
    resolution: int,
) -> str:
    """Return the LaTeX-friendly simulation count and resolution."""
    return f"{count}\\,$\\times$\\,{resolution}"


def create_table(
    dataset: SuiteDataset,
) -> str:
    rows: list[str] = []
    for _, suite_stats in dataset.items():
        input_stats = cast(
            dict[str, float],
            suite_stats["input"],
        )
        measured_stats = cast(
            dict[str, StatsBlock],
            suite_stats["measured"],
        )
        count = int(input_stats["count"])
        resolution = int(input_stats["Nres"])
        viscosity = float(input_stats["nu"])
        target_mach = input_stats["target_Mach"]
        target_reynolds_number = input_stats["target_Re"]
        alpha_nl_stats = measured_stats["log10_alpha_nl"]
        gamma_exp_stats = measured_stats["log10_gamma_exp_times_t0"]
        nl_duration_stats = measured_stats["log10_nl_duration_normed_by_t0"]
        nl_exponent_stats = measured_stats["p_nl"]
        mach_str = str(target_mach)
        reynolds_number_str = str(target_reynolds_number)
        viscosity_str = create_scientific_value_string(
            value=viscosity,
            num_significant_digits=2,
        )
        simulations_str = create_simulations_at_resolution_string(
            count=count,
            resolution=resolution,
        )
        gamma_exp_str = create_log10_stats_string(gamma_exp_stats)
        alpha_nl_str = create_log10_stats_string(alpha_nl_stats)
        nl_exponent_str = create_linear_stats_string(nl_exponent_stats)
        nl_duration_str = create_duration_stats_string(nl_duration_stats)
        rows.append(
            f"{mach_str} & {reynolds_number_str} & {viscosity_str} & {simulations_str} & {gamma_exp_str} & {alpha_nl_str} & {nl_exponent_str} & {nl_duration_str} \\\\",
        )
    return "\n\n".join(rows)


##
## === MAIN PROGRAM
##


def main() -> None:
    script_dir = Path(__file__).parent
    dataset_path = (script_dir / ".." / ".." / "datasets" / "suite_scalings.json").resolve()
    dataset = cast(
        SuiteDataset,
        json_io.read_json_file_into_dict(
            dataset_path,
            verbose=False,
        ),
    )

    def compute_sort_key(
        item: tuple[str, SuiteStats],
    ) -> tuple[float, float, int]:
        suite_stats = item[1]
        input_stats = cast(
            dict[str, float],
            suite_stats["input"],
        )
        mach = float(input_stats["target_Mach"])
        reynolds_number = float(input_stats["target_Re"])
        resolution = int(input_stats["Nres"])
        return (mach, reynolds_number, resolution)

    sorted_dataset = dict(
        sorted(
            dataset.items(),
            key=compute_sort_key,
        ),
    )
    table_tex = create_table(sorted_dataset)
    print(table_tex)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
