## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################

import numpy
import emcee
from tqdm import tqdm
from pathlib import Path
from scipy.stats import gaussian_kde
from . import plot_chain_evolution
from . import plot_model_posteriors
from . import plot_model_fits


## ###############################################################
## BASE ROUTINE
## ###############################################################

class BaseMCMCRoutine:

  def _model(self, param_vector: tuple[float, ...]):
    raise NotImplementedError()

  def _check_params_are_valid(self, param_vector: tuple[float, ...], print_errors: bool = False):
    raise NotImplementedError()

  def _get_output_params(self) -> tuple[numpy.ndarray, list[str]]:
    return self.fitted_posterior_samples, self.fitted_param_labels

  def _annotate_fitted_params(self, axs):
    pass

  def _annotate_output_params(self, axs):
    pass

  def __init__(
      self,
      *,
      output_directory    : str | Path,
      routine_name        : str,
      x_values            : list | numpy.ndarray,
      y_values            : list | numpy.ndarray,
      initial_params      : tuple[float, ...],
      prior_kde           = None,
      likelihood_sigma    : float = 1.0,
      y_data_label        : str | None = None,
      fitted_param_labels : list[str] = [],
      verbose             : bool = True,
      debug_mode          : bool = False,
    ):
    self.output_directory    = output_directory
    self.routine_name        = routine_name
    self.x_values            = numpy.asarray(x_values)
    self.y_values            = numpy.asarray(y_values)
    self.initial_params      = initial_params
    self.num_params          = len(self.initial_params)
    self.prior_kde           = prior_kde
    self.likelihood_sigma    = likelihood_sigma
    self.y_data_label        = y_data_label
    self.fitted_param_labels = fitted_param_labels
    self.verbose             = verbose
    self.debug_mode          = debug_mode
    self._validate_inputs()
    self.fitted_posterior_samples = None
    self.fitted_posterior_kde     = None
    self.output_posterior_samples = None
    self.output_posterior_kde     = None

  def _validate_inputs(self):
    if not isinstance(self.x_values, (list, numpy.ndarray)):
      raise ValueError(f"`x_values` should be either a list or array of values.")
    if not isinstance(self.y_values, (list, numpy.ndarray)):
      raise ValueError(f"`y_values` should be either a list or array of values.")
    if len(self.x_values) != len(self.y_values):
      raise ValueError(f"`x_values` and `y_values` should be the same length, but got {len(self.x_values)} vs {len(self.y_values)}.")
    if not isinstance(self.initial_params, tuple):
      raise ValueError(f"`initial_params` must be a tuple, got {type(self.initial_params)}.")
    if not isinstance(self.likelihood_sigma, float):
      raise ValueError(f"`likelihood_sigma` should be a scalar.")

  def sample_posterior(
      self,
      num_walkers   : int = 100,
      num_steps     : int = 3000,
      burn_in_steps : int = 1000,
    ):
    if not self._check_params_are_valid(self.initial_params, print_errors=True):
      raise ValueError("Initial guess is invalid!")
    print("Estimating the posterior...")
    self.num_walkers = num_walkers
    perturbed_params = numpy.array(self.initial_params) + 1e-4 * numpy.random.randn(self.num_walkers, self.num_params)
    mcmc_sampler = emcee.EnsembleSampler(self.num_walkers, self.num_params, self._log_posterior)
    for _ in tqdm(
      mcmc_sampler.sample(perturbed_params, iterations=num_steps),
      total = num_steps
      ):
      pass
    self.raw_chain = mcmc_sampler.get_chain()
    self.fitted_posterior_samples = mcmc_sampler.get_chain(discard=burn_in_steps, thin=10, flat=True)
    self.output_posterior_samples, self.output_param_labels = self._get_output_params()
    if numpy.array_equal(self.output_posterior_samples, self.fitted_posterior_samples):
      print("Estimating the KDE of only the fitted posterior...")
      self.fitted_posterior_kde = gaussian_kde(self.fitted_posterior_samples.T, bw_method="scott")
      self.output_posterior_kde = self.fitted_posterior_kde
    else:
      print("Estimating the KDE of both the fitted and output posteriors...")
      self.fitted_posterior_kde = gaussian_kde(self.fitted_posterior_samples.T, bw_method="scott")
      self.output_posterior_kde = gaussian_kde(self.output_posterior_samples.T, bw_method="scott")
    self._make_plots()

  def _log_posterior(self, param_vector):
    lp_value = self._log_prior(param_vector)
    if not numpy.isfinite(lp_value): return -numpy.inf
    ll_value = self._log_likelihood(param_vector)
    return lp_value + ll_value

  def _log_prior(self, param_vector):
    if not self._check_params_are_valid(param_vector):
      return -numpy.inf
    if self.prior_kde is not None:
      kde_vector = self._get_kde_eval_params(param_vector)
      lp_value = self.prior_kde.logpdf(kde_vector.reshape(-1, 1))[0]
      return lp_value
    return 0

  def _get_kde_eval_params(self, param_vector: tuple[float, ...]) -> numpy.ndarray:
    return numpy.asarray(param_vector)

  def _log_likelihood(self, param_vector):
    if not self._check_params_are_valid(param_vector):
      return -numpy.inf
    try:
      residual = self.y_values - self._model(param_vector)
      ll_value = -0.5 * numpy.sum(numpy.square(residual / self.likelihood_sigma))
      if not numpy.isfinite(ll_value):
        return -numpy.inf
      return ll_value
    except Exception as error:
      print("Error in likelihood:", error, param_vector)
      return -numpy.inf

  def _make_plots(self):
    plot_chain_evolution.PlotChainEvolution(self).plot()
    plot_model_posteriors.PlotModelPosteriors(self).plot()
    plot_model_fits.PlotModelFits(self).plot()


## END OF MODULE