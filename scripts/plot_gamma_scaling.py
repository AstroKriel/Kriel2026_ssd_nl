import numpy
from pathlib import Path
from jormi.ww_io import io_manager, json_files
from jormi.ww_data import fit_data
from jormi.ww_plots import plot_manager, plot_data, add_annotations, add_color


def extract_key_param_samples(fitted_posterior_samples):
  init_energy_samples    = 10**fitted_posterior_samples[:, 0]
  sat_energy_samples     = 10**fitted_posterior_samples[:, 1]
  gamma_samples          = fitted_posterior_samples[:, 2]
  start_nl_time_samples  = fitted_posterior_samples[:, 3]
  start_sat_time_samples = fitted_posterior_samples[:, 4]
  start_nl_energy        = init_energy_samples * numpy.exp(gamma_samples * start_nl_time_samples)
  alpha_samples          = (sat_energy_samples - start_nl_energy) / (start_sat_time_samples - start_nl_time_samples)
  return gamma_samples, alpha_samples, sat_energy_samples

def main():
  base_directory = Path("../data/").resolve()
  data_directories = io_manager.ItemFilter(
    include_string = ["Mach", "Re", "Pm", "Nres"]
  ).filter(
    directory = base_directory
  )
  fig, ax = plot_manager.create_figure()
  cmap_Mach, norm_Mach = add_color.create_cmap(
    cmap_name = "cmr.watermelon",
    vmin      = numpy.log10(0.1),
    vmid      = 0,
    vmax      = numpy.log10(5),
  )
  for data_directory in data_directories:
    sim_data_path = io_manager.combine_file_path_parts([ data_directory, "dataset.json" ])
    sim_data_dict = json_files.read_json_file_into_dict(sim_data_path, verbose=False)
    fit_data_path = io_manager.combine_file_path_parts([ data_directory, "stage2_fitted_posterior_samples.npy" ])
    if not io_manager.does_file_exist(fit_data_path): continue
    print(f"Loading: {data_directory}")
    fitted_posterior_samples = numpy.load(fit_data_path)
    gamma_samples, _, _ = extract_key_param_samples(fitted_posterior_samples)
    gamma_p16, gamma_p50, gamma_p84 = numpy.percentile(gamma_samples, [16, 50, 84])
    gamma_err_lower = gamma_p50 - gamma_p16
    gamma_err_upper = gamma_p84 - gamma_p50
    Mach_number = sim_data_dict["plasma_params"]["Mach"]
    Re_number = sim_data_dict["plasma_params"]["Re"]
    Mach_color = cmap_Mach(norm_Mach(numpy.log10(Mach_number)))
    ax.errorbar(
      Re_number,
      gamma_p50 / Mach_number,
      yerr = [
        [gamma_err_lower / Mach_number],
        [gamma_err_upper / Mach_number],
      ],
      fmt="o", color=Mach_color, markersize=5, capsize=3, zorder=3
    )
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlim([1e2, 1e4])
  ax.set_ylim([1e-1, 3])
  x_values = numpy.logspace(1, 5, 100)
  plot_data.plot_wo_scaling_axis(
    ax       = ax,
    x_values = x_values,
    y_values = 3e-2 * x_values**(1/2),
    ls       = "--",
    lw       = 1.5
  )
  plot_data.plot_wo_scaling_axis(
    ax       = ax,
    x_values = x_values,
    y_values = 1e-2 * x_values**(1/2),
    ls       = ":",
    lw       = 1.5
  )
  rotation = fit_data.get_line_angle_in_box(
    slope               = 1/2,
    domain_bounds       = (2, 4, numpy.log10(1e-1), numpy.log10(3)),
    domain_aspect_ratio = 6/4,
  )
  add_annotations.add_text(
    ax          = ax,
    x_pos       = 0.05,
    y_pos       = 0.45,
    label       = r"$3 \times 10^{-2}\, \mathrm{Re}^{1/2}$",
    x_alignment = "left",
    y_alignment = "bottom",
    rotate_deg  = rotation
  )
  add_annotations.add_text(
    ax          = ax,
    x_pos       = 0.95,
    y_pos       = 0.575,
    label       = r"$2.5 \times 10^{-3}\, \mathrm{Re}^{1/2}$",
    x_alignment = "right",
    y_alignment = "top",
    rotate_deg  = rotation
  )
  ax.set_xlabel(r"$\mathrm{Re}$")
  ax.set_ylabel(r"$(t_\mathrm{turb} / \ell_\mathrm{turb}) \gamma$")
  add_color.add_cbar_from_cmap(
    ax    = ax,
    cmap  = cmap_Mach,
    norm  = norm_Mach,
    label = r"$\log_{10}(\mathcal{M})$",
    side  = "top",
  )
  plot_manager.save_figure(fig, "gamma_scaling.png")


if __name__ == "__main__":
  main()


## end