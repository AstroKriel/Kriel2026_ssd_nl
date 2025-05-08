## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from pathlib import Path
from jormi.utils import list_utils
from jormi.ww_io import io_manager, json_files
from jormi.ww_plots import plot_manager
from jormi.parallelism import independent_tasks
import mcmc_routine
import my_utils


## ###############################################################
## FINAL ENERGY MODEL
## ###############################################################
def energy_model(time, stage1_params, stage2_params):
  (log10_init_energy, _, gamma) = stage1_params
  (start_nl_time, start_sat_time, log10_sat_energy) = stage2_params
  ## mask different ssd phases
  mask_exp_phase = time < start_nl_time
  mask_nl_phase  = (start_nl_time <= time) & (time < start_sat_time)
  mask_sat_phase = start_sat_time < time
  ## calculate model constants
  init_energy     = 10**log10_init_energy
  start_nl_energy = init_energy * numpy.exp(gamma * start_nl_time)
  sat_energy      = 10**log10_sat_energy
  alpha           = (sat_energy - start_nl_energy) / (start_sat_time - start_nl_time)
  ## model energy evolution
  energy = numpy.zeros_like(time)
  energy[mask_exp_phase] = init_energy * numpy.exp(gamma * time[mask_exp_phase])
  energy[mask_nl_phase]  = start_nl_energy + alpha * (time[mask_nl_phase] - start_nl_time)
  energy[mask_sat_phase] = sat_energy
  return energy


## ###############################################################
## STAGE 1 MCMC FITTER
## ###############################################################
class Stage1MCMC(mcmc_routine.BaseMCMCModel):
  def __init__(self, output_directory, time, measured, verbose):
    self.log10_e = numpy.log10(numpy.exp(1))
    super().__init__(
      output_directory = output_directory,
      routine_name     = "stage1",
      verbose          = verbose,
      time             = time,
      measured         = numpy.log10(measured),
      param_guess      = (-20, 0.85 * numpy.max(time), 0.5),
      param_labels     = [
        r"$\log_{10}(E_{\mathrm{init}})$",
        r"$t_{\mathrm{approx}}$",
        r"$\gamma$"
      ]
    )

  def _model(self, fit_params):
    (log10_init_energy, transition_time, gamma) = fit_params
    ## mask into two rough phases
    mask_exp = self.time < transition_time
    mask_sat = ~mask_exp
    ## model log_10 energy evolution
    log10_energy = numpy.zeros_like(self.time)
    log10_energy[mask_exp] = log10_init_energy + self.log10_e * gamma * self.time[mask_exp]
    log10_energy[mask_sat] = log10_init_energy + self.log10_e * gamma * transition_time
    return log10_energy

  def _check_params_are_valid(self, fit_params, print_errors=False):
    (log10_init_energy, transition_time, gamma) = fit_params
    errors = []
    if not (-30 < log10_init_energy < -5):
      errors.append(f"`log10_init_energy` ({log10_init_energy:.2f}) must be between -20 and -5.")
    if not (0.25 * self.max_time < transition_time < 0.9 * self.max_time):
      errors.append(f"`transition_time` ({transition_time:.2f}) must be between 25 and 90 percent of `max_time` ({self.max_time:.2f}).")
    if not (0 < gamma < 1):
      errors.append(f"`gamma` ({gamma:.2f}) must be between 0 and 1.")
    if len(errors) > 0:
      if print_errors: print("\n".join(errors))
      return False
    return True

  def _plot_model_results(self, fit_params):
    (_, transition_time, gamma) = fit_params
    fig, axs = plot_manager.create_figure(num_rows=3, share_x=True)
    data_args = dict(color="blue", marker="o", ms=5, ls="-", lw=1.0, zorder=3)
    measured_dlog10y_dt = numpy.gradient(self.measured, self.time)
    axs[0].plot(self.time, self.measured, **data_args)
    axs[1].plot(self.time, measured_dlog10y_dt, **data_args)
    model_args = dict(color="red", ls="-", lw=1.5, zorder=5)
    modelled   = self._model(fit_params)
    residuals  = self.measured - modelled
    axs[0].plot(self.time, modelled, **model_args)
    axs[1].axhline(y=self.log10_e * gamma, color="red", ls="--", lw=1.5)
    axs[2].plot(self.time, residuals, **model_args)
    for row_index in range(len(axs)):
      axs[row_index].axvline(x=transition_time, color="red", ls="--", lw=1.5)
    axs[1].axhline(y=0.0, color="black", ls="--")
    axs[2].axhline(y=0.0, color="black", ls="--")
    axs[0].set_ylabel(r"$\log_{10}(E_{\rm mag})$")
    axs[1].set_ylabel(r"$({\rm d}/{\rm d}t) \log_{10}(E_{\rm mag})$")
    axs[2].set_ylabel(r"residuals")
    axs[2].set_xlabel("t")
    fig_name = f"{self.routine_name}_fit.png"
    fig_file_path = io_manager.combine_file_path_parts([ self.output_directory, fig_name ])
    plot_manager.save_figure(fig, fig_file_path, verbose=self.verbose)


## ###############################################################
## STAGE 2 MCMC FITTER
## ###############################################################
class Stage2MCMC(mcmc_routine.BaseMCMCModel):
  def __init__(self, output_directory, time, measured, stage1_params, verbose):
    log10_init_energy, self.transition_time, self.gamma = stage1_params
    self.init_energy       = 10**(log10_init_energy)
    log10_sat_energy_guess = numpy.log10(self.init_energy * numpy.exp(self.gamma * self.transition_time))
    transition_index       = list_utils.get_index_of_closest_value(time, self.transition_time)
    super().__init__(
      output_directory = output_directory,
      routine_name = "stage2",
      verbose      = verbose,
      time         = time,
      measured     = measured,
      # smooth_sigma = 3.0,
      ll_sigma     = numpy.std(measured[transition_index:]),
      param_guess  = (0.85 * self.transition_time, 1.25 * self.transition_time, log10_sat_energy_guess),
      param_labels = [
        r"$t_{\mathrm{nl}}$",
        r"$t_{\mathrm{sat}}$",
        r"$\log_{10}(E_{\mathrm{sat}})$"
      ]
    )

  def _model(self, fit_params):
    (start_nl_time, start_sat_time, log10_sat_energy) = fit_params
    ## mask different ssd phases
    mask_exp_phase = self.time < start_nl_time
    mask_nl_phase  = (start_nl_time <= self.time) & (self.time < start_sat_time)
    mask_sat_phase = start_sat_time < self.time
    ## calculate model constants
    start_nl_energy = self.init_energy * numpy.exp(self.gamma * start_nl_time)
    sat_energy      = 10**log10_sat_energy
    alpha           = (sat_energy - start_nl_energy) / (start_sat_time - start_nl_time)
    ## model energy evolution
    energy = numpy.zeros_like(self.time)
    energy[mask_exp_phase] = self.init_energy * numpy.exp(self.gamma * self.time[mask_exp_phase])
    energy[mask_nl_phase]  = start_nl_energy + alpha * (self.time[mask_nl_phase] - start_nl_time)
    energy[mask_sat_phase] = sat_energy
    return energy

  def _check_params_are_valid(self, fit_params, print_errors=False):
    (start_nl_time, start_sat_time, log10_sat_energy) = fit_params
    errors = []
    if not (0.1 * self.max_time < start_nl_time < self.transition_time):
      errors.append(f"`start_nl_time` ({start_nl_time:.2f}) must be larger than 0.1 * `max_time` ({self.max_time:.2f}) and smaller than the stage-1 estimated transition time ({self.transition_time:.2f}).")
    if not (self.transition_time < start_sat_time < self.max_time):
      errors.append(f"`start_sat_time` ({start_sat_time:.2f}) must be larger than the stage-1 estimated transition time ({self.transition_time:.2f}) and less than `max_time` ({self.max_time:.2f}).")
    if not (-5 < log10_sat_energy < 0):
      errors.append(f"`log10_sat_energy` ({log10_sat_energy:.2f}) must be between -5 and 0.")
    if len(errors) > 0:
      if print_errors: print("\n".join(errors))
      return False
    return True

  def _plot_model_results(self, fit_params):
    (start_nl_time, start_sat_time, log10_sat_energy) = fit_params
    fig, axs  = plot_manager.create_figure(num_rows=3, share_x=True)
    data_args = dict(color="blue", marker="o", ms=5, ls="-", lw=1.0, zorder=3)
    measured_dy_dt = numpy.gradient(self.measured, self.time)
    axs[0].plot(self.time, self.measured, **data_args)
    axs[1].plot(self.time, measured_dy_dt, **data_args)
    model_args      = dict(color="red", ls="-", lw=1.5, zorder=5)
    start_nl_energy = self.init_energy * numpy.exp(self.gamma * start_nl_time)
    sat_energy      = 10**log10_sat_energy
    alpha           = (sat_energy - start_nl_energy) / (start_sat_time - start_nl_time)
    modelled        = self._model(fit_params)
    residual        = (self.measured - modelled) / self.ll_sigma
    axs[0].plot(self.time, modelled, **model_args)
    axs[1].axhline(y=alpha, color="red", ls="--", lw=1.5)
    axs[2].plot(self.time, residual, **model_args)
    for row_index in range(len(axs)):
      ax = axs[row_index]
      ax.axvline(x=start_nl_time, color="red", ls="--", lw=1.5)
      ax.axvline(x=start_sat_time, color="red", ls="--", lw=1.5)
      ax.axhline(y=0.0, color="black", ls="--")
    axs[0].set_ylabel(r"$E_{\rm mag}$")
    axs[1].set_ylabel(r"$({\rm d}/{\rm d}t) E_{\rm mag}$")
    axs[2].set_ylabel(r"residuals")
    axs[-1].set_xlabel("t")
    fig_name = f"{self.routine_name}_fit.png"
    fig_file_path = io_manager.combine_file_path_parts([ self.output_directory, fig_name ])
    plot_manager.save_figure(fig, fig_file_path, verbose=self.verbose)


## ###############################################################
## MCMC ROUTINE
## ###############################################################
def routine(sim_directory, level1_output_directory, verbose=True):
  sim_name = my_utils.get_sim_name(sim_directory)
  fig_file_name  = f"{sim_name}_fit.png"
  fig_file_path  = io_manager.combine_file_path_parts([ level1_output_directory, fig_file_name ])
  json_file_name = f"{sim_name}_params.json"
  json_file_path = io_manager.combine_file_path_parts([ level1_output_directory, json_file_name ])
  level2_output_directory = io_manager.combine_file_path_parts([ level1_output_directory, sim_name ])
  io_manager.init_directory(level2_output_directory, verbose=False)
  ## load and interpolate data
  data_dict = my_utils.load_data(sim_directory, num_samples=70)
  time = data_dict["time"]
  measured_energy = data_dict["magnetic_energy"]
  ## stage 1 MCMC fitter
  stage1_mcmc = Stage1MCMC(
    output_directory = level2_output_directory,
    time             = time,
    measured         = measured_energy,
    verbose          = verbose
  )
  stage1_params = stage1_mcmc.estimate_params()
  if verbose: stage1_mcmc.print_log_likelihood(stage1_params)
  ## stage 2 MCMC fitter
  stage2_mcmc = Stage2MCMC(
    output_directory = level2_output_directory,
    time             = time,
    measured         = measured_energy,
    stage1_params    = stage1_params,
    verbose          = verbose
  )
  stage2_params = stage2_mcmc.estimate_params()
  if verbose: stage2_mcmc.print_log_likelihood(stage2_params)
  ## plot final fit
  fig, axs = plot_manager.create_figure(num_rows=2, share_x=True)
  fig_data_params = dict(color="blue", marker="o", ms=3, lw=1)
  fig_fit_params  = dict(color="red", lw=2)
  modelled_energy = energy_model(time, stage1_params, stage2_params)
  axs[0].plot(data_dict["time"], data_dict["magnetic_energy"], **fig_data_params)
  axs[1].plot(data_dict["time"], numpy.log10(data_dict["magnetic_energy"]), **fig_data_params)
  axs[0].plot(data_dict["time"], modelled_energy, **fig_fit_params)
  axs[1].plot(data_dict["time"], numpy.log10(modelled_energy), **fig_fit_params)
  axs[0].set_ylabel(r"$\mathrm{energy}$")
  axs[1].set_ylabel(r"$\log_{10}(\mathrm{energy})$")
  axs[1].set_xlabel(r"time")
  plot_manager.save_figure(fig, fig_file_path, verbose=verbose)
  fit_params_dict = {"stage1_params": stage1_params, "stage2_params": stage2_params}
  json_files.save_dict_to_json_file(
    file_path  = json_file_path,
    input_dict = fit_params_dict,
    overwrite  = True,
    verbose    = verbose
  )


## ###############################################################
## FIT ALL SSD SIMULATIONS
## ###############################################################
def run_all_sims_in_parallel(output_directory):
  file_names_in_output_directory = [
    file.name
    for file in output_directory.iterdir()
    if file.is_file()
  ]
  sim_directories = sorted(Path("/scratch").glob("*/nk7952/Re*/Mach*/Pm1/576*"))
  sim_directories = [
    sim_directory
    for sim_directory in sim_directories
    if f"{my_utils.get_sim_name(sim_directory)}_params.json" in file_names_in_output_directory # recompute all
  ]
  print("Will be looking at all of the following:")
  [
    print(str(sim_directory))
    for sim_directory in sim_directories
  ]
  print(" ")
  args_list = [
    (sim_directory, output_directory, False)
    for sim_directory in sim_directories
  ]
  independent_tasks.run_in_parallel(
    func            = routine,
    args_list       = args_list,
    num_procs       = 4,
    timeout_seconds = 5 * 60,
    show_progress   = True
  )


## ###############################################################
## ONLY FIT A SINGLE SIMULATION
## ###############################################################
def run_single_sim(output_directory):
  sim_directory = "/scratch/jh2/nk7952/Re500/Mach0.1/Pm1/576v4"
  routine(sim_directory, output_directory)


## ###############################################################
## PROGRAM MAIN
## ###############################################################
def main():
  script_directory = io_manager.get_caller_directory()
  output_directory = io_manager.combine_file_path_parts([ script_directory, "fits" ])
  io_manager.init_directory(output_directory, verbose=False)
  run_all_sims_in_parallel(output_directory)
  # run_single_sim(output_directory)


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF SCRIPT