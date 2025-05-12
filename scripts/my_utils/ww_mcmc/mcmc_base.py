## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
import emcee
from pathlib import Path
from jormi.ww_io import io_manager
from jormi.utils import list_utils
from jormi.ww_data import compute_stats
from jormi.ww_plots import plot_manager


## ###############################################################
## ROUTINE
## ###############################################################
class BaseMCMCModel:
  def __init__(
      self,
      output_directory : str | Path,
      routine_name     : str,
      x_data           : list | numpy.ndarray,
      y_data           : list | numpy.ndarray,
      param_guess      : tuple[float, ...],
      ll_sigma         : float | list | numpy.ndarray = 1.0,
      param_labels     : list[str] = [],
      verbose          : bool = True
    ):
    self.output_directory = output_directory
    self.routine_name     = routine_name
    self.x_data           = numpy.asarray(x_data)
    self.y_data           = numpy.asarray(y_data)
    self.ll_sigma         = ll_sigma
    self.param_guess      = param_guess
    self.param_labels     = param_labels
    self.verbose          = verbose
    self._validate_inputs()

  def _model(self, fit_params: tuple[float, ...]):
    raise NotImplementedError()

  def _check_params_are_valid(self, fit_params: tuple[float, ...], print_errors: bool = False):
    raise NotImplementedError()

  def _plot_model_results(self, fit_params: tuple[float, ...]):
    raise NotImplementedError()

  def _validate_inputs(self):
    if not isinstance(self.x_data, (list, numpy.ndarray)):
      raise ValueError(f"`x_data` should be either a list or array of values.")
    if not isinstance(self.y_data, (list, numpy.ndarray)):
      raise ValueError(f"`y_data` should be either a list or array of values.")
    if len(self.x_data) != len(self.y_data):
      raise ValueError(f"`x_data` and `y_data` should be the same length, but got {len(self.x_data)} vs {len(self.y_data)}.")
    if not isinstance(self.ll_sigma, (float, int)):
      raise ValueError(f"`ll_sigma` should be a scalar.")
    self.ll_sigma = float(self.ll_sigma)

  def estimate_params(
      self,
      num_walkers   : int = 200,
      num_steps     : int = 5000,
      burn_in_steps : int = 2000,
      num_samples   : int = 1000,
      plot_guess    : bool = False,
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
    chain           = sampler.get_chain()
    samples         = sampler.get_chain(discard=burn_in_steps, thin=10, flat=True)
    best_fit_params = self._get_best_fit(sampler, burn_in_steps)
    self._plot_chain_evolution(chain)
    self._plot_param_estimates(samples)
    self._plot_model_results(best_fit_params)

  def _sample_posterior_distribution(self, samples, num_samples):
    indices = numpy.random.choice(len(samples), size=num_samples, replace=True)
    return samples[indices]

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
      residual = self.y_data - self._model(fit_params)
      ll_value = -0.5 * numpy.sum(numpy.square(residual / self.ll_sigma))
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
    for param_index in range(num_params):
      for walker_index in range(num_walkers):
        axs[param_index].plot(chain[:, walker_index, param_index], alpha=0.3, lw=0.5)
      axs[param_index].set_ylabel(self.param_labels[param_index])
    axs[-1].set_xlabel("steps")
    fig_name = f"{self.routine_name}_chain_evolution.png"
    fig_file_path = io_manager.combine_file_path_parts([ self.output_directory, fig_name ])
    plot_manager.save_figure(fig, fig_file_path, verbose=self.verbose)

  def _plot_param_estimates(self, samples):
    _, num_params = samples.shape
    fig, axs = plot_manager.create_figure(
      num_cols   = num_params,
      num_rows   = num_params,
      axis_shape = (5,5)
    )
    param_mins = []
    param_maxs = []
    for param1_index in range(num_params):
      for param2_index in range(num_params):
        ax = axs[param1_index, param2_index]
        if param1_index == param2_index:
          param_min, param_max = self._plot_pdf(ax, samples, param1_index)
          param_mins.append(param_min)
          param_maxs.append(param_max)
        elif param1_index > param2_index:
          self._plot_jpdf(ax, samples, param1_index, param2_index)
        else: ax.axis("off")
    for param1_index in range(num_params):
      for param2_index in range(num_params):
        ax = axs[param1_index, param2_index]
        if param1_index == param2_index:
          ax.set_xlim(param_mins[param1_index], param_maxs[param1_index])
        elif param1_index > param2_index:
          ax.set_xlim(param_mins[param2_index], param_maxs[param2_index])
          ax.set_ylim(param_mins[param1_index], param_maxs[param1_index])
        if param1_index == num_params-1: ax.set_xlabel(self.param_labels[param2_index])
        if param2_index == 0: ax.set_ylabel(self.param_labels[param1_index])
    fig_name = f"{self.routine_name}_corner_plot.png"
    fig_file_path = io_manager.combine_file_path_parts([ self.output_directory, fig_name ])
    plot_manager.save_figure(fig, fig_file_path, verbose=self.verbose)

  def _plot_pdf(self, ax, samples, param_index):
    values = samples[:, param_index]
    bin_centers, estimated_pdf = compute_stats.estimate_pdf(values=values, num_bins=20)
    ax.step(bin_centers, estimated_pdf, where="mid", lw=2, color="black")
    p16 = numpy.percentile(values, 16)
    p50 = numpy.percentile(values, 50)
    p84 = numpy.percentile(values, 84)
    label = f"{self.param_labels[param_index]} $= {p50:.2f}_{{-{p50-p16:.2f}}}^{{+{p84-p50:.2f}}}$"
    ax.set_title(label, pad=15)
    if param_index > 0: ax.tick_params(labelleft=False, labelright=True)
    if param_index < samples.shape[1]-1: ax.set_xticklabels([])
    threshold_value = 0.05 * numpy.max(estimated_pdf)
    index_lower = list_utils.find_first_crossing(values=estimated_pdf, target=threshold_value, direction="rising")
    index_upper = list_utils.find_first_crossing(values=estimated_pdf, target=threshold_value, direction="falling")
    bin_lower = bin_centers[index_lower]
    bin_upper = bin_centers[index_upper]
    return (bin_lower, bin_upper)

  def _plot_jpdf(self, ax, samples, param1_index, param2_index):
    row_data = samples[:, param1_index]
    col_data = samples[:, param2_index]
    bc_rows, bc_cols, estimated_jpdf = compute_stats.estimate_jpdf(data_x=col_data, data_y=row_data, num_bins=50)
    extent = [ bc_cols[0], bc_cols[-1], bc_rows[0], bc_rows[-1] ]
    ax.imshow(
      estimated_jpdf,
      origin = "lower",
      extent = extent,
      aspect = "auto",
      cmap   = "Blues"
    )
    if param1_index < samples.shape[1]-1: ax.set_xticklabels([])
    if param2_index > 0: ax.set_yticklabels([])

  # def _plot_predictive_band(self, posterior_samples, confidence=0.95, num_points=100):
  #   all_predictions = []
  #   x_eval = numpy.linspace(self.x_data.min(), self.x_data.max(), num_points)
  #   for params in posterior_samples:
  #       y_pred = self._model(params) if len(self.x_data) == num_points else self._model(params, x_eval)
  #       all_predictions.append(y_pred)

  #   all_predictions = numpy.array(all_predictions)
  #   lower_bound = numpy.percentile(all_predictions, (1 - confidence) / 2 * 100, axis=0)
  #   upper_bound = numpy.percentile(all_predictions, (1 + confidence) / 2 * 100, axis=0)
  #   median_pred = numpy.median(all_predictions, axis=0)

  #   import matplotlib.pyplot as plt
  #   plt.fill_between(x_eval, lower_bound, upper_bound, color="skyblue", alpha=0.4, label=f"{int(confidence*100)}% credible band")
  #   plt.plot(x_eval, median_pred, color="blue", label="Median prediction")
  #   plt.scatter(self.x_data, self.y_data, color="black", s=10, label="Observed")
  #   plt.xlabel("x")
  #   plt.ylabel("y")
  #   plt.legend()
  #   plt.title("Posterior Predictive Fit with Credible Interval")
  #   plt.show()


## END OF MODULE