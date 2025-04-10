## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from scipy.ndimage import gaussian_filter1d as scipy_filter1d
from loki.ww_plots import plot_manager
from loki.ww_data import interpolate_data
import utils # local


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
  fig, axs = plot_manager.create_figure(num_rows=3, share_x=True, axis_shape=(6, 10))
  num_points_raw    = 100
  num_points_interp = 50
  time_bounds       = [0, 100]
  time_transition   = 50
  time_data_raw     = utils.generate_nonuniform_domain(domain_bounds=time_bounds, num_points=num_points_raw)
  energy_data_raw   = utils.generate_data(x_data=time_data_raw, noise_level=3.0, x_transition=time_transition)
  _time_data_interp = utils.generate_uniform_domain(domain_bounds=time_bounds, num_points=num_points_interp) # could be truncated in the next step
  time_data_interp, energy_data_interp = interpolate_data.interpolate_1d(time_data_raw, energy_data_raw, _time_data_interp, kind="linear")
  axs[0].plot(time_data_raw, energy_data_raw, color="blue", marker="o", ms=5, zorder=3, label="raw data")
  plot_data(
    axs         = axs,
    time_data   = time_data_interp,
    energy_data = energy_data_interp,
    color       = "red",
    label       = "uniformly sampled",
  )
  plot_data(
    axs         = axs,
    time_data   = time_data_interp,
    energy_data = scipy_filter1d(energy_data_interp, 1.0),
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
  axs[1].set_ylim([-20, 20])
  axs[2].set_ylim([-5, 5])
  plot_manager.save_figure(fig, "estimate_using_gradients.png")



## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF SCRIPT