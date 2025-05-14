## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
import emcee
from pathlib import Path
from scipy.stats import gaussian_kde
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
    self.samples = None

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
      plot_guess    : bool = False,
    ):
    if not self._check_params_are_valid(self.param_guess, print_errors=True):
      raise ValueError("Initial guess is invalid!")
    if plot_guess:
      self._plot_model_results(self.param_guess)
      return self.param_guess
    print("Estimating parameters...")
    self.num_walkers = num_walkers
    self.num_params = len(self.param_guess)
    param_positions = numpy.array(self.param_guess) + 1e-4 * numpy.random.randn(self.num_walkers, self.num_params)
    sampler = emcee.EnsembleSampler(self.num_walkers, self.num_params, self._log_posterior)
    sampler.run_mcmc(param_positions, num_steps)
    self.chain   = sampler.get_chain()
    self.samples = sampler.get_chain(discard=burn_in_steps, thin=10, flat=True)
    self._compute_scaled_kde()
    self._plot_chain_evolution()
    self._plot_param_estimates()
    self._plot_model_results()

  def print_log_likelihood(self, fit_params):
    ll_value = self._log_likelihood(fit_params)
    print(f"params = ({fit_params}) yields log-likelihood = {ll_value:.2e}")

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

  def _compute_scaled_kde(self):
    print("Estimating KDE...")
    self.kde = gaussian_kde(self.samples.T, bw_method="scott")

  def _plot_chain_evolution(self):
    fig, axs = plot_manager.create_figure(num_rows=self.num_params, num_cols=1, share_x=True)
    for param_index in range(self.num_params):
      for walker_index in range(self.num_walkers):
        axs[param_index].plot(self.chain[:, walker_index, param_index], alpha=0.3, lw=0.5)
      axs[param_index].set_ylabel(self.param_labels[param_index])
    axs[-1].set_xlabel("steps")
    fig_name = f"{self.routine_name}_chain_evolution.png"
    fig_file_path = io_manager.combine_file_path_parts([ self.output_directory, fig_name ])
    plot_manager.save_figure(fig, fig_file_path, verbose=self.verbose)

  def _plot_pdf(self, ax, param_index):
    values = self.samples[:, param_index]
    bin_centers, estimated_pdf = compute_stats.estimate_pdf(values=values, num_bins=20)
    ax.step(bin_centers, estimated_pdf, where="mid", lw=2, color="black")
    p16, p50, p84 = numpy.percentile(values, [16, 50, 84])
    label = f"{self.param_labels[param_index]} $= {p50:.2f}_{{-{p50-p16:.2f}}}^{{+{p84-p50:.2f}}}$"
    ax.set_title(label, pad=15)
    if param_index > 0: ax.tick_params(labelleft=False, labelright=True)
    if param_index < self.samples.shape[1]-1: ax.set_xticklabels([])
    threshold_value = 0.05 * numpy.max(estimated_pdf)
    index_lower = list_utils.find_first_crossing(values=estimated_pdf, target=threshold_value, direction="rising")
    index_upper = list_utils.find_first_crossing(values=estimated_pdf, target=threshold_value, direction="falling")
    bin_lower = bin_centers[index_lower]
    bin_upper = bin_centers[index_upper]
    return (bin_lower, bin_upper)

  def _plot_param_estimates(self):
    fig, axs = plot_manager.create_figure(
      num_cols   = self.num_params,
      num_rows   = self.num_params,
      axis_shape = (5,5)
    )
    param_mins = []
    param_maxs = []
    for row_index in range(self.num_params):
      for col_index in range(self.num_params):
        ax = axs[row_index, col_index]
        if row_index == col_index:
          param_min, param_max = self._plot_pdf(ax, row_index)
          param_mins.append(param_min)
          param_maxs.append(param_max)
        elif row_index > col_index:
          self._plot_jpdf(ax, row_index, col_index)
          # self._plot_kde(ax, row_index, col_index)
        else: ax.axis("off")
    self._label_plot(axs, param_mins, param_maxs)
    fig_name = f"{self.routine_name}_corner_plot.png"
    fig_file_path = io_manager.combine_file_path_parts([ self.output_directory, fig_name ])
    plot_manager.save_figure(fig, fig_file_path, verbose=self.verbose)

  def _label_plot(self, axs, param_mins, param_maxs):
    for row_index in range(self.num_params):
      for col_index in range(self.num_params):
        if col_index > row_index: continue
        ax = axs[row_index, col_index]
        ## diagonal: 1d pdfs
        if row_index == col_index:
          ax.set_xlim(param_mins[row_index], param_maxs[row_index])
          if col_index > 0: ax.yaxis.tick_right()
        ## lower triangle: 2d joint-pdfs
        elif row_index > col_index:
          ax.set_xlim(param_mins[col_index], param_maxs[col_index])
          ax.set_ylim(param_mins[row_index], param_maxs[row_index])
          if col_index > 0: ax.set_yticklabels([])
        ## bottom row: add x-axis labels
        if row_index == self.num_params-1:
          ax.set_xlabel(self.param_labels[col_index])
        else: ax.set_xticklabels([])
        ## first column: add y-axis labels
        if col_index == 0:
          ax.set_ylabel(self.param_labels[row_index])
        

  def _plot_jpdf(self, ax, row_index, col_index):
    row_data = self.samples[:, row_index]
    col_data = self.samples[:, col_index]
    bc_rows, bc_cols, estimated_jpdf = compute_stats.estimate_jpdf(data_x=col_data, data_y=row_data, num_bins=50)
    extent = [ bc_cols[0], bc_cols[-1], bc_rows[0], bc_rows[-1] ]
    ax.imshow(
      estimated_jpdf,
      extent = extent,
      origin = "lower",
      aspect = "auto",
      cmap   = "Blues"
    )

  def _plot_kde(self, ax, row_index, col_index):
    print(f"Estimating KDE projection: axs[{row_index}][{col_index}]")
    Xi, Xj, Z = compute_2d_kde_projection(
        full_kde = self.kde,
        samples = self.samples,
        i=col_index,
        j=row_index,
        num_points=30,
        num_marginal_samples=50,
    )
    ax.contour(Xi, Xj, Z, alpha=0.5, colors="red", linewidths=1.0, zorder=5)

def compute_2d_kde_projection(full_kde, samples, i, j, num_points, num_marginal_samples):
    ndim = samples.shape[1]
    other_indices = [k for k in range(ndim) if k != i and k != j]
    xi = numpy.linspace(samples[:, i].min(), samples[:, i].max(), num_points)
    xj = numpy.linspace(samples[:, j].min(), samples[:, j].max(), num_points)
    Xi, Xj = numpy.meshgrid(xi, xj)
    X_flat = Xi.ravel()
    Y_flat = Xj.ravel()
    n_grid = X_flat.shape[0]
    # Draw marginal samples
    marginal_values = samples[numpy.random.choice(samples.shape[0], size=num_marginal_samples)][..., other_indices]
    # Repeat grid and marginal samples to build full KDE input
    grid_points = numpy.zeros((n_grid * num_marginal_samples, ndim))
    grid_points[:, i] = numpy.repeat(X_flat, num_marginal_samples)
    grid_points[:, j] = numpy.repeat(Y_flat, num_marginal_samples)
    marginal_tiled = numpy.tile(marginal_values, (n_grid, 1))
    grid_points[:, other_indices] = marginal_tiled
    # Evaluate and reshape
    Z_vals = full_kde(grid_points.T)
    Z_avg = Z_vals.reshape(n_grid, num_marginal_samples).mean(axis=1).reshape(num_points, num_points)
    return Xi, Xj, Z_avg


## END OF MODULE