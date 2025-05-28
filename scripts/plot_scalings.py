import numpy
from pathlib import Path
from jormi.ww_io import io_manager, json_files
from jormi.ww_plots import plot_manager, plot_data, add_annotations, add_color


def plot_powerlaw_passing_through(ax, domain_bounds, coordinate, slope, num_samples, ls):
  x_values = numpy.linspace(domain_bounds[0], domain_bounds[1], num_samples)
  (x1, y1) = coordinate
  a0 = y1 / x1**slope
  y_values = a0 * x_values**slope
  plot_data.plot_wo_scaling_axis(
    ax     = ax,
    x_values = x_values,
    y_values = y_values,
    ls     = ls,
    lw     = 2
  )

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
  fig, axs = plot_manager.create_figure(num_rows=3, num_cols=2)
  cmap_Mach, norm_Mach = add_color.create_cmap(
    cmap_name="cmr.watermelon", vmin=numpy.log10(0.1), vmid=0, vmax=numpy.log10(5)
  )
  cmap_Re, norm_Re = add_color.create_cmap(
    cmap_name="Blues_r", vmin=numpy.log10(500), vmax=numpy.log10(5000), cmax=0.85
  )
  for data_directory in data_directories:
    sim_data_path = io_manager.combine_file_path_parts([ data_directory, "dataset.json" ])
    sim_data_dict = json_files.read_json_file_into_dict(sim_data_path, verbose=False)
    fit_data_path = io_manager.combine_file_path_parts([ data_directory, "stage2_fitted_posterior_samples.npy" ])
    if not io_manager.does_file_exist(fit_data_path): continue
    print(f"Loading: {data_directory}")
    fitted_posterior_samples = numpy.load(fit_data_path)
    gamma_samples, alpha_samples, sat_energy_samples = extract_key_param_samples(fitted_posterior_samples)
    gamma_p16, gamma_p50, gamma_p84 = numpy.percentile(gamma_samples, [16, 50, 84])
    gamma_err_lower = gamma_p50 - gamma_p16
    gamma_err_upper = gamma_p84 - gamma_p50
    alpha_p16, alpha_p50, alpha_p84 = numpy.percentile(alpha_samples, [16, 50, 84])
    alpha_err_lower = alpha_p50 - alpha_p16
    alpha_err_upper = alpha_p84 - alpha_p50
    sat_energy_p16, sat_energy_p50, sat_energy_p84 = numpy.percentile(sat_energy_samples, [16, 50, 84])
    sat_energy_err_lower = sat_energy_p50 - sat_energy_p16
    sat_energy_err_upper = sat_energy_p84 - sat_energy_p50
    Mach_number = sim_data_dict["plasma_params"]["Mach"]
    Re_number = sim_data_dict["plasma_params"]["Re"]
    Mach_color = cmap_Mach(norm_Mach(numpy.log10(Mach_number)))
    Re_color = cmap_Re(norm_Re(numpy.log10(Re_number)))
    axs[0,0].errorbar(
      Re_number, gamma_p50,
      yerr=[[gamma_err_lower], [gamma_err_upper]],
      fmt="o", color=Mach_color, markersize=5, capsize=3, zorder=3
    )
    axs[0,1].errorbar(
      Mach_number, gamma_p50,
      yerr=[[gamma_err_lower], [gamma_err_upper]],
      fmt="o", color=Re_color, markersize=5, capsize=3, zorder=3
    )
    axs[1,0].errorbar(
      Re_number, alpha_p50,
      yerr=[[alpha_err_lower], [alpha_err_upper]],
      fmt="o", color=Mach_color, markersize=5, capsize=3, zorder=3
    )
    axs[1,1].errorbar(
      Mach_number, alpha_p50,
      yerr=[[alpha_err_lower], [alpha_err_upper]],
      fmt="o", color=Re_color, markersize=5, capsize=3, zorder=3
    )
    axs[2,0].errorbar(
      Re_number, sat_energy_p50,
      yerr=[[sat_energy_err_lower], [sat_energy_err_upper]],
      fmt="o", color=Mach_color, markersize=5, capsize=3, zorder=3
    )
    axs[2,1].errorbar(
      Mach_number, sat_energy_p50,
      yerr=[[sat_energy_err_lower], [sat_energy_err_upper]],
      fmt="o", color=Re_color, markersize=5, capsize=3, zorder=3
    )
  for row_index, row_ax in enumerate(axs):
    row_ax[0].set_xscale("log")
    row_ax[1].set_xscale("log")
    row_ax[0].set_yscale("log")
    row_ax[1].set_yscale("log")
    row_ax[1].set_yticklabels([])
    if row_index < axs.shape[0]-1:
      row_ax[0].set_xticklabels([])
      row_ax[1].set_xticklabels([])
  axs[2,0].set_xlabel(r"$\mathrm{Re}$")
  axs[2,1].set_xlabel(r"$\mathcal{M}$")
  axs[0,0].set_ylabel(r"$\gamma$")
  axs[1,0].set_ylabel(r"$\alpha$")
  axs[2,0].set_ylabel(r"saturated energy")
  add_color.add_cbar_from_cmap(
    ax    = axs[0,0],
    cmap  = cmap_Mach,
    norm  = norm_Mach,
    label = r"$\log_{10}(\mathcal{M})$",
    side  = "top",
  )
  add_color.add_cbar_from_cmap(
    ax    = axs[0,1],
    cmap  = cmap_Re,
    norm  = norm_Re,
    label = r"$\log_{10}(\mathrm{Re})$",
    side  = "top",
  )
  ## gamma scalings
  plot_powerlaw_passing_through(
    ax            = axs[0,0],
    domain_bounds = (1e2, 1e4),
    coordinate    = (1e3, 5e-1),
    slope         = 1/2,
    num_samples   = 10,
    ls            = ":"
  )
  plot_powerlaw_passing_through(
    ax            = axs[0,1],
    domain_bounds = (1e-2, 1e1),
    coordinate    = (1e0, 1e0),
    slope         = 1,
    num_samples   = 10,
    ls            = ":"
  )
  add_annotations.add_text(
    ax    = axs[0,0],
    x_pos = 0.525,
    y_pos = 0.65,
    label = r"$\sim\mathrm{Re}^{1/2}$",
  )
  add_annotations.add_text(
    ax    = axs[0,1],
    x_pos = 0.05,
    y_pos = 0.45,
    label = r"$\sim\mathcal{M}$",
  )
  ## alpha scalings
  axs[1,0].axhline(y=1e-3, color="black", ls=":", lw=2, zorder=1)
  plot_powerlaw_passing_through(
    ax            = axs[1,1],
    domain_bounds = (1e-2, 1e1),
    coordinate    = (1e-1, 3e-6),
    slope         = 3,
    num_samples   = 10,
    ls            = ":",
  )
  add_annotations.add_text(
    ax    = axs[1,1],
    x_pos = 0.05,
    y_pos = 0.4,
    label = r"$\sim\mathcal{M}^3$",
  )
  ## sat level scalings
  axs[2,0].axhline(y=2e-2, color="black", ls=":", lw=2, zorder=1)
  plot_powerlaw_passing_through(
    ax            = axs[2,1],
    domain_bounds = (1e-2, 1e1),
    coordinate    = (1e0, 1e-1),
    slope         = 2,
    num_samples   = 10,
    ls            = ":"
  )
  add_annotations.add_text(
    ax    = axs[2,1],
    x_pos = 0.05,
    y_pos = 0.4,
    label = r"$\sim\mathcal{M}^2$",
  )
  plot_manager.save_figure(fig, "scalings.png")


if __name__ == "__main__":
  main()


## end