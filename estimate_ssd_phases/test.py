## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
import emcee
import corner
from jormi.ww_io import flash_data
from jormi.ww_data import interpolate_data
from jormi.ww_plots import plot_manager


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def check_params_are_valid(params, known_params, debug_mode=False):
  (start_nl_phase, start_sat_phase, gamma, ln_sat_energy) = params
  sat_energy = numpy.exp(ln_sat_energy)
  max_time = known_params
  errors = []
  if not (10 < start_nl_phase < 0.85 * max_time):
    errors.append(f"`start_nl_phase` ({start_nl_phase}) must be >10 and <85% of max_time.")
  if not (start_nl_phase < start_sat_phase < max_time):
    errors.append(f"`start_sat_phase` ({start_sat_phase}) must be > start_nl_phase and < max_time.")
  if not (100 * (start_sat_phase - start_nl_phase) / max_time > 5):
    errors.append(f"non-linear phase duration should be at least 5 percent of the total simulation time.")
  if not (0 < gamma < 1):
    errors.append("`gamma` should be between 0 and 1.")
  if not (-4 < numpy.log10(sat_energy) < -1):
    errors.append("`sat_energy` should be between 1e-4 and 1e-1.")
  if len(errors) > 0:
    if debug_mode: print("\n".join(errors))
    return False
  return True

def load_data(num_samples = 100):
  time, measured_energy = flash_data.read_vi_data(
    directory    = "/scratch/jh2/nk7952/Re500/Mach0.3/Pm1/576",
    dataset_name = "mag"
  )
  interp_time, interp_energy = interpolate_data.interpolate_1d(
    x_values = time[1:],
    y_values = measured_energy[1:],
    x_interp = numpy.linspace(time[1], time[-1], num_samples),
    kind     = "linear"
  )
  return interp_time, interp_energy


## ###############################################################
## MODELS
## ###############################################################
def dlny_dt_model(time, params):
  time = numpy.array(time)
  (start_nl_phase, start_sat_phase, gamma, _) = params
  ## mask different ssd phases
  mask_exp_phase = time <= start_nl_phase
  mask_nl_phase  = (start_nl_phase < time) & (time <= start_sat_phase)
  ## model
  slope   = gamma / (start_sat_phase - start_nl_phase)
  dlny_dt = numpy.zeros_like(time)
  dlny_dt[mask_exp_phase] = gamma
  dlny_dt[mask_nl_phase]  = gamma - slope * (time[mask_nl_phase] - start_nl_phase)
  return dlny_dt

def energy_model(time, params):
  time = numpy.array(time)
  (start_nl_phase, start_sat_phase, _, ln_sat_energy) = params
  sat_energy = numpy.exp(ln_sat_energy)
  ## mask different ssd phases
  mask_nl_phase  = (start_nl_phase < time) & (time <= start_sat_phase)
  mask_sat_phase = start_sat_phase < time
  ## model
  slope  = sat_energy / (start_sat_phase - start_nl_phase)
  energy = numpy.zeros_like(time)
  energy[mask_nl_phase]  = slope * (time[mask_nl_phase] - start_nl_phase)
  energy[mask_sat_phase] = sat_energy
  return energy


## ###############################################################
## MCMC OPERATOR
## ###############################################################
class MCMCModel:
  def __init__(self, time, measured_energy, known_params):
    self.time            = time
    self.measured_energy = measured_energy
    self.dlny_dt         = numpy.gradient(numpy.log(measured_energy), time)
    self.dy_dt           = numpy.gradient(measured_energy, time)
    self.known_params    = known_params

  def estimate_params(
      self,
      initial_guess,
      num_walkers   = 200,
      num_steps     = 5000,
      burn_in_steps = 2000,
      skip_mcmc     = False,
    ):
    if skip_mcmc:
      self._plot_model_results(initial_guess)
      return
    num_params = len(initial_guess)
    param_positions = numpy.array(initial_guess) + 1e-4 * numpy.random.randn(num_walkers, num_params)
    sampler = emcee.EnsembleSampler(num_walkers, num_params, self._log_posterior)
    sampler.run_mcmc(param_positions, num_steps)
    chain            = sampler.get_chain()
    samples          = sampler.get_chain(discard=burn_in_steps, thin=10, flat=True)
    estimated_params = numpy.median(samples, axis=0)
    self._plot_chain_evolution(chain)
    self._corner_plot(samples)
    self._plot_model_results(estimated_params)

  def _plot_chain_evolution(self, chain):
    _, num_walkers, num_params = chain.shape
    fig, axs = plot_manager.create_figure(
      num_rows = num_params,
      num_cols = 1,
      share_x  = True
    )
    for param_index in range(num_params):
      ax = axs[param_index]
      for walker_index in range(num_walkers):
        ax.plot(chain[:, walker_index, param_index], alpha=0.3, lw=0.5)
    axs[-1].set_xlabel("steps")
    plot_manager.save_figure(fig, "mcmc_chain_evolution.png")

  def _corner_plot(self, samples):
    fig = corner.corner(samples)
    plot_manager.save_figure(fig, "mcmc_corner_plot.png")

  def _plot_model_results(self, estimated_params):
    fig, axs = plot_manager.create_figure(
      num_rows   = 3,
      num_cols   = 2,
      share_x    = True,
      axis_shape = (6, 10),
      x_spacing  = 0.3,
      y_spacing  = 0.1,
    )
    data_args = dict(color="blue", marker="o", ms=5, ls="-", lw=1.0, zorder=3)
    axs[0,0].plot(self.time, numpy.log(self.measured_energy), **data_args)
    axs[0,1].plot(self.time, self.measured_energy, **data_args)
    axs[1,0].plot(self.time, self.dlny_dt, **data_args)
    axs[1,1].plot(self.time, self.dy_dt, **data_args)
    mcmc_args = dict(color="green", ls="-", lw=2.0, zorder=3)
    modelled_dlny_dt = dlny_dt_model(self.time, estimated_params)
    modelled_energy  = energy_model(self.time, estimated_params)
    axs[1,0].plot(self.time, modelled_dlny_dt, **mcmc_args)
    axs[0,1].plot(self.time, modelled_energy, **mcmc_args)
    for row_index in range(3):
      for col_index in range(2):
        ax = axs[row_index, col_index]
        ax.axvline(x=estimated_params[0], color="green", ls="--", lw=2.0, zorder=2)
        ax.axvline(x=estimated_params[1], color="green", ls="--", lw=2.0, zorder=2)
    axs[1,0].axhline(y=0.0, color="red", ls="--", lw=2.0, zorder=1)
    axs[1,1].axhline(y=0.0, color="red", ls="--", lw=2.0, zorder=1)
    axs[0,0].set_ylabel(r"$\ln(E_{\rm mag})$")
    axs[0,1].set_ylabel(r"$E_{\rm mag}$")
    axs[1,0].set_ylabel(r"$({\rm d}/{\rm d}t) \ln(E_{\rm mag})$")
    axs[1,1].set_ylabel(r"$({\rm d}/{\rm d}t) E_{\rm mag}$")
    axs[2,0].set_xlabel("t")
    axs[2,1].set_xlabel("t")
    plot_manager.save_figure(fig, f"mcmc_result.png")

  def _log_likelihood(self, params):
    if not check_params_are_valid(params, self.known_params):
      return -numpy.inf
    modelled_dlny_dt  = dlny_dt_model(self.time, params)
    modelled_energy   = energy_model(self.time, params)
    residuals_dlny_dt = self.dlny_dt - modelled_dlny_dt
    residuals_energy  = self.measured_energy - modelled_energy
    return -0.5 * (
      numpy.sum(numpy.square(residuals_dlny_dt)) + numpy.sum(numpy.square(residuals_energy))
    )

  def _log_prior(self, params):
    if not check_params_are_valid(params, self.known_params):
      return -numpy.inf
    return 0
  
  def _log_posterior(self, params):
    log_prior_value = self._log_prior(params)
    if not numpy.isfinite(log_prior_value): return -numpy.inf
    log_likelihood_value = self._log_likelihood(params)
    return log_prior_value + log_likelihood_value


## ###############################################################
## PROGRAM MAIN
## ###############################################################
def main():
  ## load and interpolate data
  time, measured_energy = load_data(100)
  ## estimate parameters using MCMC
  max_time      = numpy.max(time)
  known_params  = max_time
  mcmc_model    = MCMCModel(time, measured_energy, known_params)
  # initial_guess = (0.25*max_time, 0.75*max_time, 1e-1, 3e-3)
  # mcmc_model.estimate_params(initial_guess)
  my_best_guess = (125, 170, 0.2, numpy.log(6e-3))
  mcmc_model.estimate_params(my_best_guess, skip_mcmc=True)


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF SCRIPT