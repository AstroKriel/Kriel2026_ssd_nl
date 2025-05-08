## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
import emcee
import corner
from pathlib import Path
from scipy.ndimage import gaussian_filter1d as scipy_filter1d
from jormi.ww_io import io_manager
from jormi.ww_plots import plot_manager


## ###############################################################
## ROUTINE
## ###############################################################
class BaseMCMCModel:
  def __init__(
      self,
      output_directory : str | Path,
      routine_name     : str,
      time             : list | numpy.ndarray,
      measured         : list | numpy.ndarray,
      param_guess      : tuple[float, ...],
      smooth_sigma     : float | None = None,
      ll_sigma         : float | list | numpy.ndarray = 1.0,
      param_labels     : list[str] = [],
      verbose          : bool = True
    ):
    self.output_directory = output_directory
    self.routine_name     = routine_name
    self.max_time         = numpy.max(time)
    self.time             = numpy.asarray(time)
    self.measured         = numpy.asarray(measured)
    self.ll_sigma         = ll_sigma
    self.param_guess      = param_guess
    self.param_labels     = param_labels
    self.verbose          = verbose
    self._validate_inputs()
    if smooth_sigma: self.measured = scipy_filter1d(self.measured, smooth_sigma)

  def _model(self, fit_params: tuple[float, ...]):
    raise NotImplementedError()

  def _check_params_are_valid(self, fit_params: tuple[float, ...], print_errors: bool = False):
    raise NotImplementedError()

  def _plot_model_results(self, fit_params: tuple[float, ...]):
    raise NotImplementedError()
    fig, axs = plot_manager.create_figure(
      num_rows  = 3,
      num_cols  = 1,
      share_x   = True,
      y_spacing = 0.1,
      x_spacing = 0.5
    )
    plot_manager.save_figure(fig, f"{self.routine_name}_fit.png", verbose=self.verbose)

  def _validate_inputs(self):
    if len(self.time) != len(self.measured):
      raise ValueError(f"`time` and `measured` should be the same length, but got {len(self.time)} vs {len(self.measured)}.")
    if isinstance(self.ll_sigma, (float, int)):
      self.ll_sigma = float(self.ll_sigma)
    else:
      self.ll_sigma = numpy.asarray(self.ll_sigma)
      if len(self.ll_sigma) != len(self.time):
        raise ValueError(f"`ll_sigma` must be a scalar or be the same length as `time`, but got {len(self.ll_sigma)} vs {len(self.time)}.")

  def estimate_params(
      self,
      num_walkers    : int = 200,
      num_steps      : int = 5000,
      burn_in_steps  : int = 2000,
      plot_guess     : bool = False,
    ):
    if not self._check_params_are_valid(self.param_guess, print_errors=True):
      raise ValueError("Initial guess is invalid!")
    if plot_guess:
      self._plot_model_results(self.param_guess)
      return self.param_guess
    num_params      = len(self.param_guess)
    param_positions = numpy.array(self.param_guess) + 1e-4 * numpy.random.randn(num_walkers, num_params)
    sampler         = emcee.EnsembleSampler(num_walkers, num_params, self._log_posterior)
    sampler.run_mcmc(param_positions, num_steps)
    chain      = sampler.get_chain()
    samples    = sampler.get_chain(discard=burn_in_steps, thin=10, flat=True)
    fit_params = self._get_best_fit(sampler, burn_in_steps)
    self._plot_chain_evolution(chain)
    self._plot_param_space(samples)
    self._plot_model_results(fit_params)
    return fit_params

  def print_log_likelihood(self, fit_params):
    ll_value = self._log_likelihood(fit_params)
    print(f"params = ({fit_params}) yields log-likelihood = {ll_value:.2e}")

  def _get_best_fit(self, sampler, burn_in_steps):
    ## TODO: look into posterior predictive distribution check
    samples         = sampler.get_chain(discard=burn_in_steps, thin=10, flat=True)
    log_probs       = sampler.get_log_prob(discard=burn_in_steps, thin=10, flat=True)
    best_fit_index  = numpy.argmax(log_probs)
    best_fit_params = samples[best_fit_index]
    return best_fit_params

  def _log_prior(self, fit_params):
    if not self._check_params_are_valid(fit_params):
      return -numpy.inf
    return 0

  def _log_likelihood(self, fit_params):
    ## TODO: look into gaussian likelihood modelling
    if not self._check_params_are_valid(fit_params):
      return -numpy.inf
    try:
      residual    = self.measured - self._model(fit_params)
      chi_squared = numpy.sum(numpy.square(residual / self.ll_sigma))
      ll_value    = -0.5 * chi_squared
      if not numpy.isfinite(ll_value):
        return -numpy.inf
      return ll_value
    except Exception as e:
      print("Error in likelihood:", e, fit_params)
      return -numpy.inf

  def _log_posterior(self, fit_params):
    lp_value = self._log_prior(fit_params)
    if not numpy.isfinite(lp_value): return -numpy.inf
    ll_value = self._log_likelihood(fit_params)
    return lp_value + ll_value

  def _plot_chain_evolution(self, chain):
    _, num_walkers, num_params = chain.shape
    fig, axs = plot_manager.create_figure(num_rows=num_params, num_cols=1, share_x=True)
    for i in range(num_params):
      for w in range(num_walkers):
        axs[i].plot(chain[:, w, i], alpha=0.3, lw=0.5)
      axs[i].set_ylabel(self.param_labels[i])
    axs[-1].set_xlabel("steps")
    fig_name = f"{self.routine_name}_chain_evolution.png"
    fig_file_path = io_manager.combine_file_path_parts([ self.output_directory, fig_name ])
    plot_manager.save_figure(fig, fig_file_path, verbose=self.verbose)

  def _plot_param_space(self, samples):
    fig = corner.corner(samples, labels=self.param_labels)
    fig_name = f"{self.routine_name}_corner_plot.png"
    fig_file_path = io_manager.combine_file_path_parts([ self.output_directory, fig_name ])
    plot_manager.save_figure(fig, fig_file_path, verbose=self.verbose)


## END OF MODULE