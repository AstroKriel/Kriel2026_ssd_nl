## ###############################################################
## DEPENDANCIES
## ###############################################################
import sys
import numpy
from pathlib import Path
from jormi.ww_io import io_manager, csv_files
from jormi.ww_plots import plot_manager
from my_utils import ww_sims


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  script_directory = io_manager.get_caller_directory()
  output_directory = io_manager.combine_file_path_parts([ script_directory, "data" ])
  io_manager.init_directory(output_directory)
  print(" ")
  for sim_directory in sorted(Path("/scratch").glob("*/nk7952/Re500/Mach0.5/Pm1/576*")):
    sim_name    = ww_sims.get_sim_name(sim_directory)
    data_dict   = ww_sims.load_data(sim_directory)
    fig, axs    = plot_manager.create_figure(num_rows=2, share_x=True)
    plot_params = dict(color="black", marker="o", ms=3, lw=1)
    axs[0].plot(data_dict["time"], data_dict["magnetic_energy"], **plot_params)
    axs[1].plot(data_dict["time"], numpy.log10(data_dict["magnetic_energy"]), **plot_params)
    axs[0].set_ylabel("$\mathrm{energy}$")
    axs[1].set_ylabel("$\log_{10}(\mathrm{energy})$")
    axs[1].set_xlabel("time")
    fig_file_name = f"{sim_name}.png"
    fig_file_path = io_manager.combine_file_path_parts([ output_directory, fig_file_name ])
    plot_manager.save_figure(fig, fig_file_path)
    csv_file_name = f"{sim_name}.csv"
    csv_file_path = io_manager.combine_file_path_parts([ output_directory, csv_file_name ])
    csv_files.save_dict_to_csv_file(csv_file_path, data_dict, overwrite=True)
    print(" ")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit(0)


## END OF SCRIPT