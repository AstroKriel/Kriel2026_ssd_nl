## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from scipy.ndimage import gaussian_filter1d as scipy_filter1d
from jormi.ww_io import flash_data
from jormi.ww_data import interpolate_data
from jormi.ww_plots import plot_manager


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def plot_data(axs, time_data, energy_data, color, label):
  plot_style = { "color":color, "ls":"-", "lw":1.5, "marker":"o", "ms":5, "zorder":3, "label":label }
  dydt_data   = numpy.gradient(energy_data, time_data)
  d2ydt2_data = numpy.gradient(dydt_data, time_data)
  axs[0].plot(time_data, energy_data, **plot_style)
  axs[1].plot(time_data, dydt_data,   **plot_style)
  axs[2].plot(time_data, d2ydt2_data, **plot_style)


## ###############################################################
## DEMO: DERIVATIVE COMPARISONS
## ###############################################################
def main():
  fig, axs = plot_manager.create_figure(num_rows=3, share_x=True, axis_shape=(6, 10), y_spacing=0.1)
  time_raw, energy_raw = flash_data.read_vi_data(
    directory    = "/scratch/jh2/nk7952/Re1500/Mach0.1/Pm1/144",
    dataset_name = "mag",
  )
  axs[0].plot(time_raw, energy_raw, color="blue", marker="o", ms=5, zorder=3, label="raw data")
  _time_interp = numpy.linspace(numpy.min(time_raw), numpy.max(time_raw), 100)
  time_interp, energy_interp = interpolate_data.interpolate_1d(time_raw, energy_raw, _time_interp, kind="linear")
  plot_data(
    axs         = axs,
    time_data   = time_interp,
    energy_data = energy_interp,
    color       = "red",
    label       = "uniformly sampled",
  )
  plot_data(
    axs         = axs,
    time_data   = time_interp,
    energy_data = scipy_filter1d(energy_interp, 2.0),
    color       = "green",
    label       = "uniformly sampled + smoothed",
  )
  axs[0].set_ylabel("y-values")
  axs[1].set_ylabel("first derivatives")
  axs[2].set_ylabel("second derivatives")
  axs[2].set_xlabel("x-values")
  axs[0].legend(loc="lower right")
  axs[1].axhline(y=0, ls="--", color="black", zorder=1)
  axs[2].axhline(y=0, ls="--", color="black", zorder=1)
  plot_manager.save_figure(fig, "estimate_using_gradients.png")



## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF SCRIPT