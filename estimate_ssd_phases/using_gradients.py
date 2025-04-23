## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from scipy.ndimage import gaussian_filter1d as scipy_filter1d
from jormi.utils import list_utils
from jormi.ww_io import flash_data
from jormi.ww_data import interpolate_data
from jormi.ww_plots import plot_manager, plot_data


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def plot_values(axs, time_data, energy_data, color, label=None):
  plot_style = { "color":color, "ls":"-", "lw":1.5, "marker":"o", "ms":5, "zorder":3, "label":label }
  dydt_data   = numpy.gradient(energy_data, time_data)
  d2ydt2_data = numpy.gradient(dydt_data, time_data)
  axs[0].plot(time_data, energy_data, **plot_style)
  axs[1].plot(time_data, dydt_data,   **plot_style)
  axs[2].plot(time_data, d2ydt2_data, **plot_style)
  return dydt_data, d2ydt2_data


## ###############################################################
## DEMO: DERIVATIVE COMPARISONS
## ###############################################################
def main():
  # start_nl_time  = 45
  # end_nl_time    = 60
  # start_nl_time  = 9
  # end_nl_time    = 12
  start_nl_time  = 18
  end_nl_time    = 25
  # start_nl_time  = None
  # end_nl_time    = None
  fig, axs = plot_manager.create_figure(
    num_rows   = 3,
    num_cols   = 2,
    share_x    = True,
    axis_shape = (6, 10),
    x_spacing  = 0.3,
    y_spacing  = 0.1,
  )
  time_raw, energy_raw = flash_data.read_vi_data(
    # directory    = "/scratch/jh2/nk7952/Re1500/Mach0.5/Pm1/576",
    # directory    = "/scratch/jh2/nk7952/Re1500/Mach5/Pm1/576",
    directory    = "/scratch/jh2/nk7952/Re1500/Mach2/Pm1/576",
    dataset_name = "mag",
    start_time   = 1.0,
  )
  log10_energy_raw = numpy.log10(energy_raw)
  axs[0,0].plot(time_raw, log10_energy_raw, color="blue", marker="o", ms=5, zorder=3, label="raw data")
  axs[0,1].plot(time_raw, energy_raw, color="blue", marker="o", ms=5, zorder=3, label="raw data")
  time_interp, log10_energy_interp = interpolate_data.interpolate_1d(
    x_values = time_raw,
    y_values = log10_energy_raw,
    x_interp = numpy.linspace(numpy.min(time_raw), numpy.max(time_raw), 80),
    kind     = "linear"
  )
  log10_energy_interp_filtered = scipy_filter1d(log10_energy_interp, 2.0)
  dlog10y_dt, d2log10y_dt2 = plot_values(
    axs         = axs[:,0],
    time_data   = time_interp,
    energy_data = log10_energy_interp_filtered,
    color       = "red",
    label       = r"sampling + filtering"
  )
  energy_interp_filtered = numpy.power(10, log10_energy_interp_filtered)
  dy_dt, d2y_dt2 = plot_values(
    axs         = axs[:,1],
    time_data   = time_interp,
    energy_data = energy_interp_filtered,
    color       = "red",
  )
  if (start_nl_time is not None) and (end_nl_time is not None):
    log10_init_energy = log10_energy_interp_filtered[0]
    init_energy = energy_interp_filtered[0]
    start_nl_index = list_utils.get_index_of_closest_value(time_interp, start_nl_time)
    end_nl_index   = list_utils.get_index_of_closest_value(time_interp, end_nl_time)
    ave_gamma = numpy.median(dlog10y_dt[:start_nl_index])
    std_gamma = numpy.std(dlog10y_dt[:start_nl_index])
    ave_alpha_linear = numpy.median(dy_dt[start_nl_index:end_nl_index])
    std_alpha_linear = numpy.std(dy_dt[start_nl_index:end_nl_index])
    axs[1,0].axhline(y=ave_gamma, ls=":", color="green")
    axs[1,0].fill_between(
      time_interp[:start_nl_index],
      ave_gamma - std_gamma,
      ave_gamma + std_gamma,
      color = "green",
      alpha = 0.25
    )
    plot_data.plot_wo_scaling_axis(
      ax     = axs[0,0],
      x_data = time_interp,
      y_data = log10_init_energy + ave_gamma * time_interp,
      ls     = ":",
      lw     = 2.0,
      color  = "green"
    )
    axs[1,1].axhline(y=ave_alpha_linear, ls=":", color="green")
    axs[1,1].fill_between(
      time_interp[start_nl_index+1:end_nl_index],
      ave_alpha_linear - std_alpha_linear,
      ave_alpha_linear + std_alpha_linear,
      color = "green",
      alpha = 0.25
    )
    plot_data.plot_wo_scaling_axis(
      ax     = axs[0,1],
      x_data = time_interp,
      y_data = init_energy * numpy.exp(ave_gamma * start_nl_time) + ave_alpha_linear * (time_interp - start_nl_time),
      ls     = ":",
      lw     = 2.0,
      color  = "green"
    )
    for row_index in range(3):
      for col_index in range(2):
        ax = axs[row_index, col_index]
        ax.axvline(x=start_nl_time, ls="--", color="black")
        ax.axvline(x=end_nl_time, ls="--", color="black")
  axs[0,0].set_ylabel(r"$\log_{10}(E_{\rm mag})$")
  axs[1,0].set_ylabel(r"${\rm d} \log_{10}(E_{\rm mag}) / {\rm d} t$")
  axs[2,0].set_ylabel(r"${\rm d^2} \log_{10}(E_{\rm mag}) / {\rm d} t^2$")
  axs[2,0].set_xlabel(r"$t$")
  axs[1,0].axhline(y=0, ls="--", color="black")
  axs[2,0].axhline(y=0, ls="--", color="black")
  axs[0,0].legend(loc="lower right")
  axs[0,1].set_ylabel(r"$E_{\rm mag}$")
  axs[1,1].set_ylabel(r"${\rm d} E_{\rm mag} / {\rm d} t$")
  axs[2,1].set_ylabel(r"${\rm d^2} E_{\rm mag} / {\rm d} t^2$")
  axs[2,1].set_xlabel(r"$t$")
  axs[1,1].axhline(y=0, ls="--", color="black")
  axs[2,1].axhline(y=0, ls="--", color="black")
  plot_manager.save_figure(fig, "estimate_using_gradients.png")



## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF SCRIPT