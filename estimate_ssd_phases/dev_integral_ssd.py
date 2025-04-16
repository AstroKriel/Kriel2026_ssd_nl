import numpy
from scipy.integrate import cumulative_trapezoid
from jormi.ww_io import flash_data
from jormi.ww_plots import plot_manager

def main():
  fig, axs = plot_manager.create_figure(num_rows=2, share_x=True, axis_shape=(6, 10), y_spacing=0.1)
  time_raw, energy_raw = flash_data.read_vi_data(
    directory    = "/scratch/jh2/nk7952/Re1500/Mach0.1/Pm1/144",
    dataset_name = "mag",
  )
  energy_raw = numpy.log10(energy_raw + 1e-20)
  axs[0].plot(time_raw, energy_raw, color="blue", marker="o", ms=5, zorder=3, label="raw data")
  # Remove NaNs or sort if needed (depends on your data structure)
  # Compute cumulative integral using trapezoidal rule
  energy_integrated = cumulative_trapezoid(energy_raw, time_raw, initial=0)
  axs[1].plot(time_raw, energy_integrated, color="green", lw=2, label="integrated energy")
  axs[0].set_ylabel("$\log_{10}$(energy)")
  axs[1].set_ylabel("integrated $\log_{10}$(energy)")
  axs[1].set_xlabel("time")
  plot_manager.save_figure(fig, "dev_integrated_energy.png")

if __name__ == "__main__":
  main()

