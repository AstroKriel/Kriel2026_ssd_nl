## ###############################################################
## DEPENDENCIES
## ###############################################################

import numpy
import argparse
from pathlib import Path
from jormi.utils import list_utils
from jormi.ww_io import io_manager, json_files
import my_mcmc_routine


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################

def compute_median_params_from_kde(kde, num_samples=10000):
  samples = kde.resample(num_samples)
  return tuple(numpy.median(samples, axis=1))


## ###############################################################
## PROGRAM MAIN
## ###############################################################

def main():
  parser = argparse.ArgumentParser(description="Run MCMC fitting routine.")
  parser.add_argument("-data_directory", type=str, required=True)
  data_directory = Path(parser.parse_args().data_directory).resolve()
  data_path = io_manager.combine_file_path_parts([ data_directory, "dataset.json" ])
  ## read in energy evolution
  data_dict = json_files.read_json_file_into_dict(data_path)
  x_values = data_dict["interp_data"]["time"]
  y_values = data_dict["interp_data"]["magnetic_energy"]
  ## build initial guess for stage 1
  stage1_initial_params = (
    -20, # log10(E_init)
    0.5, # log10(E_sat)
    0.5 * numpy.max(x_values) # gammma
  )
  ## run stage 1 fitter
  stage1_mcmc = my_mcmc_routine.MCMCStage1Routine(
    output_directory   = data_directory,
    x_values           = x_values,
    y_values           = y_values,
    initial_params     = stage1_initial_params,
    plot_posterior_kde = False
  )
  stage1_mcmc.estimate_posterior()
  ## extract key outputs from stage 1
  stage1_median_transition_time = numpy.median(stage1_mcmc.fitted_posterior_samples[:,2])
  stage2_prior_kde = stage1_mcmc.output_posterior_kde
  ## build initial guess for stage 2
  stage1_median_output_params = compute_median_params_from_kde(stage2_prior_kde)
  stage2_initial_params = (
    stage1_median_output_params[0], # log10(E_init)
    stage1_median_output_params[1], # log10(E_sat)
    stage1_median_output_params[2], # gammma
    0.5 * stage1_median_transition_time, # t_nl
    0.5 * (numpy.max(x_values) + stage1_median_transition_time) # t_sat
  )
  approx_transition_index = list_utils.get_index_of_closest_value(x_values, stage1_median_transition_time)
  stage2_likelihood_sigma = numpy.std(y_values[approx_transition_index:])
  ## run stage 2 fitter
  stage2_mcmc = my_mcmc_routine.MCMCStage2Routine(
    output_directory   = data_directory,
    x_values           = x_values,
    y_values           = y_values,
    initial_params     = stage2_initial_params,
    prior_kde          = stage2_prior_kde,
    likelihood_sigma   = stage2_likelihood_sigma,
    plot_posterior_kde = True
  )
  stage2_mcmc.estimate_posterior()
  my_mcmc_routine.plot_final_fits.PlotFinalFits(stage2_mcmc).plot()


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################

if __name__ == "__main__":
  main()


## END OF SCRIPT