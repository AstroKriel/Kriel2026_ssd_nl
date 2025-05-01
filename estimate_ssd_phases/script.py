## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
import emcee
import corner
from scipy.ndimage import gaussian_filter1d as scipy_filter1d
from jormi.utils import list_utils
from jormi.ww_data import interpolate_data
from jormi.ww_plots import plot_manager
from ww_flash_sims.sim_io import read_vi_data


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def load_data(num_samples = 100):
  time, measured_energy = read_vi_data.read_vi_data(
    directory    = "/scratch/jh2/nk7952/Re500/Mach0.3/Pm1/576",
    dataset_name = "mag"
  )
  interp_time, interp_energy = interpolate_data.interpolate_1d(
    x_values = time[1:],
    y_values = measured_energy[1:],
    x_interp = numpy.linspace(time[1], time[-1], num_samples),
    kind     = "linear"
  )
  return interp_time[3:], interp_energy[3:]

def gaussian(x, center, sigma):
  return numpy.exp(-0.5 * numpy.square((x - center) / sigma))


## ###############################################################
## STAGE 1 MCMC FITTER
## ###############################################################
class Stage1:
  def __init__(self, time, measured_energy):
    self.time                  = time
    self.max_time              = numpy.max(self.time)
    self.measured_log10_energy = numpy.log10(measured_energy)
    self.param_labels          = [
      r"$\log_{10}(E_{\mathrm{init}})$",
      r"$t_{\mathrm{approx}}$",
      r"$\gamma$"
    ]

  def estimate_params(
      self,
      initial_params,
      num_walkers   = 200,
      num_steps     = 5000,
      burn_in_steps = 2000,
      plot_guess    = False,
    ):
    if not self._check_params_are_valid(initial_params, print_errors=True):
      raise ValueError("Inital guess is invalid!")
    if plot_guess:
      self._plot_model_results(initial_params)
      return
    num_params = len(initial_params)
    param_positions = numpy.array(initial_params) + 1e-4 * numpy.random.randn(num_walkers, num_params)
    sampler = emcee.EnsembleSampler(num_walkers, num_params, self._log_posterior)
    sampler.run_mcmc(param_positions, num_steps)
    chain      = sampler.get_chain()
    samples    = sampler.get_chain(discard=burn_in_steps, thin=10, flat=True)
    fit_params = numpy.median(samples, axis=0)
    self._plot_chain_evolution(chain)
    self._corner_plot(samples)
    self._plot_model_results(fit_params)
    return fit_params

  def _log10_energy_model(self, fit_params):
    (log10_init_energy, transition_time, gamma) = fit_params
    mask_exp = self.time < transition_time
    mask_sat = ~mask_exp
    log10_energy = numpy.zeros_like(self.time)
    log10_energy[mask_exp] = log10_init_energy + numpy.log10(numpy.exp(1)) * gamma * self.time[mask_exp]
    log10_energy[mask_sat] = log10_init_energy + numpy.log10(numpy.exp(1)) * gamma * transition_time
    return log10_energy

  def _check_params_are_valid(self, fit_params, print_errors=False):
    (log10_init_energy, transition_time, gamma) = fit_params
    errors = []
    if not (-30 < log10_init_energy < -5):
      errors.append(f"`log10_init_energy` ({log10_init_energy:.2f}) must be beteeen -20 and -5.")
    if not (0.25 * self.max_time < transition_time < 0.9 * self.max_time):
      errors.append(f"`transition_time` ({transition_time:.2f}) must be beteeen 25 and 90 percent of `max_time` ({self.max_time:.2f}).")
    if not (0 < gamma < 1):
      errors.append(f"`gamma` ({gamma:.2f}) must be between 0 and 1.")
    if len(errors) > 0:
      if print_errors: print("\n".join(errors))
      return False
    return True

  def _log_prior(self, fit_params):
    if not self._check_params_are_valid(fit_params):
      return -numpy.inf
    return 0

  def _log_likelihood(self, fit_params, print_value=False):
    if not self._check_params_are_valid(fit_params):
        return -numpy.inf
    try:
        residual = self.measured_log10_energy - self._log10_energy_model(fit_params)
        ll_value = -0.5 * numpy.sum(numpy.square(residual))
        if print_value: print(ll_value)
        if not numpy.isfinite(ll_value):
            return -numpy.inf
        return ll_value
    except Exception as e:
        print("Error in likelihood:", e, fit_params)
        return -numpy.inf

  def _log_posterior(self, fit_params):
    log_prior_value = self._log_prior(fit_params)
    if not numpy.isfinite(log_prior_value): return -numpy.inf
    log_likelihood_value = self._log_likelihood(fit_params)
    return log_prior_value + log_likelihood_value

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
      ax.set_ylabel(self.param_labels[param_index])
    axs[-1].set_xlabel("steps")
    plot_manager.save_figure(fig, "mcmc_stage_1_chain_evolution.png")

  def _corner_plot(self, samples):
    fig = corner.corner(samples, labels=self.param_labels)
    plot_manager.save_figure(fig, "mcmc_stage_1_corner_plot.png")

  def _plot_model_results(self, fit_params):
    (_, transition_time, gamma) = fit_params
    fig, axs = plot_manager.create_figure(num_rows=3, share_x=True)
    data_args = dict(color="blue", marker="o", ms=5, ls="-", lw=1.0, zorder=3)
    measured_dlog10y_dt = numpy.gradient(self.measured_log10_energy, self.time)
    axs[0].plot(self.time, self.measured_log10_energy, **data_args)
    axs[1].plot(self.time, measured_dlog10y_dt, **data_args)
    model_args = dict(color="red", ls="-", lw=1.5, zorder=5)
    modelled_log10_energy = self._log10_energy_model(fit_params)
    residuals = self.measured_log10_energy - modelled_log10_energy
    axs[0].plot(self.time, modelled_log10_energy, **model_args)
    axs[1].axhline(y=gamma * numpy.log10(numpy.exp(1)), color="red", ls="--", lw=1.5)
    axs[2].plot(self.time, residuals, **model_args)
    for row_index in range(len(axs)):
      axs[row_index].axvline(x=transition_time, color="red", ls="--", lw=1.5)
    axs[1].axhline(y=0.0, color="black", ls="--")
    axs[2].axhline(y=0.0, color="black", ls="--")
    axs[0].set_ylabel(r"$\log_{10}(E_{\rm mag})$")
    axs[1].set_ylabel(r"$({\rm d}/{\rm d}t) \log_{10}(E_{\rm mag})$")
    axs[2].set_ylabel(r"residuals")
    axs[2].set_xlabel("t")
    plot_manager.save_figure(fig, f"mcmc_stage_1_fit.png")


## ###############################################################
## STAGE 2 MCMC FITTER
## ###############################################################
class Stage2:
  def __init__(self, time, measured_energy, stage1_params):
    (log10_init_energy, self.transition_time, self.gamma) = stage1_params
    self.time                = time
    self.max_time            = numpy.max(self.time)
    self.measured_energy     = scipy_filter1d(measured_energy, 3.0)
    self.measured_dy_dt      = numpy.gradient(self.measured_energy, self.time)
    self.init_energy         = 10**log10_init_energy
    transition_index         = list_utils.get_index_of_closest_value(self.time, self.transition_time)
    safe_sat_start_index     = int(transition_index + 0.25 * (len(self.time) - transition_index))
    self.measured_sat_energy = numpy.median(self.measured_energy[safe_sat_start_index:])
    self.param_labels        = [
      r"$t_{\mathrm{nl}}$",
      r"$t_{\mathrm{sat}}$"
    ]

  def estimate_params(
      self,
      initial_params,
      num_walkers   = 200,
      num_steps     = 5000,
      burn_in_steps = 2000,
      plot_guess    = False,
    ):
    if not self._check_params_are_valid(initial_params, print_errors=True):
      raise ValueError("Inital guess is invalid!")
    if plot_guess:
      self._plot_model_results(initial_params)
      return initial_params
    num_params = len(initial_params)
    param_positions = numpy.array(initial_params) + 1e-4 * numpy.random.randn(num_walkers, num_params)
    sampler = emcee.EnsembleSampler(num_walkers, num_params, self._log_posterior)
    sampler.run_mcmc(param_positions, num_steps)
    chain      = sampler.get_chain()
    samples    = sampler.get_chain(discard=burn_in_steps, thin=10, flat=True)
    flat_chain = sampler.get_chain(discard=burn_in_steps, thin=10, flat=True)
    log_probs  = sampler.get_log_prob(discard=burn_in_steps, thin=10, flat=True)
    max_idx    = numpy.argmax(log_probs)
    fit_params = flat_chain[max_idx]
    self._plot_chain_evolution(chain)
    self._corner_plot(samples)
    self._plot_model_results(fit_params)
    return fit_params

  def _energy_model(self, fit_params):
    (start_nl_time, start_sat_time) = fit_params
    ## mask different ssd phases
    mask_exp_phase = self.time < start_nl_time
    mask_nl_phase  = (start_nl_time <= self.time) & (self.time < start_sat_time)
    mask_sat_phase = start_sat_time < self.time
    ## intermediate values
    start_nl_energy = self.init_energy * numpy.exp(self.gamma * start_nl_time)
    alpha = (self.measured_sat_energy - start_nl_energy) / (start_sat_time - start_nl_time)
    ## model
    energy = numpy.zeros_like(self.time)
    energy[mask_exp_phase] = self.init_energy * numpy.exp(self.gamma * self.time[mask_exp_phase])
    energy[mask_nl_phase]  = start_nl_energy + alpha * (self.time[mask_nl_phase] - start_nl_time)
    energy[mask_sat_phase] = self.measured_sat_energy
    return energy

  def _check_params_are_valid(self, fit_params, print_errors=False):
    (start_nl_time, start_sat_time) = fit_params
    errors = []
    if not (0.1 * self.max_time < start_nl_time < self.transition_time):
      errors.append(f"`start_nl_time` ({start_nl_time:.2f}) must be larger than 0.1 * `max_time` ({self.max_time:.2f}) and smaller than the stage-1 estimated transition time ({self.transition_time:.2f}).")
    if not (self.transition_time < start_sat_time < self.max_time):
      errors.append(f"`start_sat_time` ({start_sat_time:.2f}) must be larger than the stage-1 estimated transition time ({self.transition_time:.2f}) and less than `max_time` ({self.max_time:.2f}).")
    if len(errors) > 0:
      if print_errors: print("\n".join(errors))
      return False
    return True

  def _log_prior(self, fit_params):
    if not self._check_params_are_valid(fit_params):
      return -numpy.inf
    return 0

  def _log_likelihood(self, fit_params, print_value=False):
    if not self._check_params_are_valid(fit_params):
        return -numpy.inf
    try:
        residual    = self.measured_energy - self._energy_model(fit_params)
        sigma       = 0.5 * self.measured_sat_energy
        chi_squared = numpy.sum(numpy.square(residual / sigma))
        ll_value    = -0.5 * chi_squared
        if print_value: print(f"params = ({fit_params[0]:.2f}, {fit_params[1]:.2f}) yields chi^2 = {chi_squared:.2e}")
        if not numpy.isfinite(ll_value):
            return -numpy.inf
        return ll_value
    except Exception as e:
        print("Error in likelihood:", e, fit_params)
        return -numpy.inf

  def plot_log_likelihood(self):
    start_sat_times = numpy.linspace(150, 250, 10)
    ll_values = []
    for start_sat_time in start_sat_times:
      fit_params = (125, start_sat_time) # (125, 170)
      ll_value = self._log_likelihood(fit_params)
      ll_values.append(ll_value)
    fig, ax = plot_manager.create_figure()
    ax.plot(start_sat_times, ll_values)
    ax.set_xlabel(self.param_labels[1])
    ax.set_ylabel("log-likelihood")
    plot_manager.save_figure(fig, "log_likelihood.png")

  def _log_posterior(self, fit_params):
    log_prior_value = self._log_prior(fit_params)
    if not numpy.isfinite(log_prior_value): return -numpy.inf
    log_likelihood_value = self._log_likelihood(fit_params)
    return log_prior_value + log_likelihood_value

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
      ax.set_ylabel(self.param_labels[param_index])
    axs[-1].set_xlabel("steps")
    plot_manager.save_figure(fig, "mcmc_stage_2_chain_evolution.png")

  def _corner_plot(self, samples):
    fig = corner.corner(samples, labels=self.param_labels)
    plot_manager.save_figure(fig, "mcmc_stage_2_corner_plot.png")

  def _plot_model_results(self, fit_params):
    (start_nl_time, start_sat_time) = fit_params
    fig, axs = plot_manager.create_figure(
      num_rows  = 2,
      num_cols  = 2,
      share_x   = True,
      y_spacing = 0.1,
      x_spacing = 0.5
    )
    ax11_right = axs[1,1].twinx()
    data_args = dict(color="blue", marker="o", ms=5, ls="-", lw=1.0, zorder=3)
    measured_dy_dt = numpy.gradient(self.measured_energy, self.time)
    axs[0,0].plot(self.time, self.measured_energy, **data_args)
    axs[0,1].plot(self.time, measured_dy_dt, **data_args)
    model_args = dict(color="red", ls="-", lw=1.5, zorder=5)
    start_nl_energy = self.init_energy * numpy.exp(self.gamma * start_nl_time)
    alpha           = (self.measured_sat_energy - start_nl_energy) / (start_sat_time - start_nl_time)
    modelled_energy = self._energy_model(fit_params)
    residual_energy = self.measured_energy - modelled_energy
    axs[0,0].plot(self.time, modelled_energy, **model_args)
    axs[0,1].axhline(y=alpha, color="red", ls="--", lw=1.5)
    axs[1,0].plot(self.time, residual_energy, **model_args)
    for row_index in range(2):
      for col_index in range(2):
        ax = axs[row_index, col_index]
        ax.axvline(x=start_nl_time, color="red", ls="--", lw=1.5)
        ax.axvline(x=start_sat_time, color="red", ls="--", lw=1.5)
    axs[0,1].axhline(y=0.0, color="black", ls="--")
    axs[1,0].axhline(y=0.0, color="black", ls="--")
    axs[1,1].axhline(y=0.0, color="black", ls="--")
    axs[0,0].set_ylabel(r"$E_{\rm mag}$")
    axs[0,1].set_ylabel(r"$({\rm d}/{\rm d}t) E_{\rm mag}$")
    axs[1,0].set_ylabel(r"residuals")
    axs[1,0].set_xlabel("t")
    axs[1,1].set_xlabel("t")
    plot_manager.save_figure(fig, f"mcmc_stage_2_fit.png")


## ###############################################################
## PROGRAM MAIN
## ###############################################################
def main():
  ## load and interpolate data
  time, measured_energy = load_data(70)
  ## stage 1 MCMC fitter
  stage1_guess  = (-20, 0.85 * numpy.max(time), 0.5)
  stage1_mcmc   = Stage1(time, measured_energy)
  stage1_params = stage1_mcmc.estimate_params(stage1_guess)
  stage1_mcmc._log_likelihood(stage1_params, print_value=True)
  ## fit using mcmc routine
  trans_time    = stage1_params[1]
  stage2_guess  = (0.85 * trans_time, 1.25 * trans_time)
  stage2_mcmc   = Stage2(time, measured_energy, stage1_params)
  stage2_params = stage2_mcmc.estimate_params(stage2_guess)
  stage2_mcmc._log_likelihood(stage2_params, print_value=True)


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF SCRIPT