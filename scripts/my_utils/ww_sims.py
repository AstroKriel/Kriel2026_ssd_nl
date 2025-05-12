## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from pathlib import Path
from jormi.ww_data import interpolate_data
from ww_flash_sims.sim_io import read_vi_data


## ###############################################################
## FUNCTIONS
## ###############################################################
def load_data(
    directory   : str | Path,
    num_samples : int = 100
  ) -> tuple[numpy.ndarray, numpy.ndarray]:
  time, magnetic_energy = read_vi_data.read_vi_data(directory=directory, dataset_name="mag")
  interp_time, interp_magnetic_energy = interpolate_data.interpolate_1d(
    x_values = time[1:],
    y_values = magnetic_energy[1:],
    x_interp = numpy.linspace(time[1], time[-1], num_samples),
    kind     = "linear"
  )
  return {"time": interp_time, "magnetic_energy": interp_magnetic_energy}

def get_sim_name(sim_directory):
  sim_path_parts = str(sim_directory).split("/")
  Re_index = [
    part_index
    for part_index, part_name in enumerate(sim_path_parts)
    if "Re" in part_name
  ][0]
  sim_name = "".join(sim_path_parts[Re_index:])
  return sim_name


## END OF MODULE