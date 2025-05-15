## ###############################################################
## DEPENDENCIES
## ###############################################################

import numpy
from jormi.ww_plots import plot_manager
from jormi.ww_io import io_manager
from jormi.ww_data import compute_stats
from jormi.utils import list_utils
from . import base_plotter


## ###############################################################
## HELPER FUNCTION
## ###############################################################



## ###############################################################
## PLOTTING ROUTINE
## ###############################################################

class PlotModelPosteriors(base_plotter.BaseMCMCPlotter):

  def plot(self):
    num_params = self.mcmc_routine.num_params
    fig, axs = plot_manager.create_figure(num_cols=num_params, num_rows=num_params, axis_shape=(5,5))
    param_mins = []
    param_maxs = []
    for row_index in range(num_params):
      for col_index in range(num_params):
        ax = axs[row_index, col_index]
        if row_index == col_index:
          param_min, param_max = self._plot_pdf(ax, row_index)
          param_mins.append(param_min)
          param_maxs.append(param_max)
        elif row_index > col_index:
          self._plot_jpdf(ax, row_index, col_index)
          self._plot_kde(ax, row_index, col_index)
        else: ax.axis("off")
    self._tweak_plot(axs, param_mins, param_maxs)
    fig_name = f"{self.mcmc_routine.routine_name}_corner_plot.png"
    file_path = io_manager.combine_file_path_parts([ self.mcmc_routine.output_directory, fig_name ])
    plot_manager.save_figure(fig, file_path, verbose=self.mcmc_routine.verbose)

  def _plot_pdf(self, ax, param_index):
    values = self.mcmc_routine.posterior_samples[:, param_index]
    bin_centers, pdf = compute_stats.estimate_pdf(values=values, num_bins=20)
    ax.step(bin_centers, pdf, where="mid", lw=2, color="black")
    p16, p50, p84 = numpy.percentile(values, [16, 50, 84])
    label = f"{self.mcmc_routine.param_labels[param_index]} $= {p50:.2f}_{{-{p50-p16:.2f}}}^{{+{p84-p50:.2f}}}$"
    ax.set_title(label, pad=15)
    if param_index > 0: ax.tick_params(labelleft=False, labelright=True)
    if param_index < self.mcmc_routine.posterior_samples.shape[1] - 1: ax.set_xticklabels([])
    pdf_threshold = 0.05 * numpy.max(pdf)
    i1 = list_utils.find_first_crossing(values=pdf, target=pdf_threshold, direction="rising")
    # i2 = list_utils.find_first_crossing(values=pdf, target=threshold, direction="falling")
    reversed_pdf = pdf[::-1]
    i2_rev = list_utils.find_first_crossing(values=reversed_pdf, target=pdf_threshold, direction="falling")
    i2 = len(pdf) - 1 - i2_rev if i2_rev is not None else None
    return bin_centers[i1], bin_centers[i2]

  def _plot_jpdf(self, ax, row_idx, col_idx):
    bc_rows, bc_cols, jpdf = compute_stats.estimate_jpdf(
      data_x   = self.mcmc_routine.posterior_samples[:, col_idx],
      data_y   = self.mcmc_routine.posterior_samples[:, row_idx],
      num_bins = 50
    )
    ax.imshow(
      jpdf,
      origin = "lower",
      aspect = "auto",
      cmap   = "Blues",
      extent = [
        bc_cols[0], bc_cols[-1],
        bc_rows[0], bc_rows[-1],
      ],
    )

  def _tweak_plot(self, axs, param_mins, param_maxs):
    for row_index in range(self.mcmc_routine.num_params):
      for col_index in range(self.mcmc_routine.num_params):
        if col_index > row_index: continue
        ax = axs[row_index, col_index]
        if row_index == col_index:
          # ax.set_xlim(param_mins[row_index], param_maxs[row_index])
          if col_index > 0: ax.yaxis.tick_right()
        else:
          # ax.set_xlim(param_mins[col_index], param_maxs[col_index])
          # ax.set_ylim(param_mins[row_index], param_maxs[row_index])
          if col_index > 0: ax.set_yticklabels([])
        if row_index == self.mcmc_routine.num_params - 1:
          ax.set_xlabel(self.mcmc_routine.param_labels[col_index])
        else: ax.set_xticklabels([])
        if col_index == 0:
          ax.set_ylabel(self.mcmc_routine.param_labels[row_index])

  def _plot_kde(self, ax, row_index, col_index):
    print(f"Estimating KDE projection for axs[{row_index}][{col_index}]")
    Xi, Xj, Z = self._compute_2d_kde_projection(
      col_index            = col_index,
      row_index            = row_index,
      num_points           = 30,
      num_marginal_samples = 50,
    )
    ax.contour(Xi, Xj, Z, colors="red", linewidths=1.0, alpha=0.5)

  def _compute_2d_kde_projection(self, col_index, row_index, num_points, num_marginal_samples):
    other_indices = [
      k
      for k in range(self.mcmc_routine.num_params)
      if k != col_index and k != row_index
    ]
    xi = numpy.linspace(
      self.mcmc_routine.posterior_samples[:, col_index].min(),
      self.mcmc_routine.posterior_samples[:, col_index].max(),
      num_points
    )
    xj = numpy.linspace(
      self.mcmc_routine.posterior_samples[:, row_index].min(),
      self.mcmc_routine.posterior_samples[:, row_index].max(),
      num_points
    )
    Xi, Xj = numpy.meshgrid(xi, xj)
    X_flat = Xi.ravel()
    Y_flat = Xj.ravel()
    n_grid = X_flat.shape[0]
    marginal_sample_indices = numpy.random.choice(
      self.mcmc_routine.posterior_samples.shape[0],
      size    = num_marginal_samples,
      replace = False
    )
    marginal_values = self.mcmc_routine.posterior_samples[marginal_sample_indices][:, other_indices]
    grid_points = numpy.zeros((n_grid * num_marginal_samples, self.mcmc_routine.num_params))
    grid_points[:, col_index] = numpy.repeat(X_flat, num_marginal_samples)
    grid_points[:, row_index] = numpy.repeat(Y_flat, num_marginal_samples)
    marginal_tiled = numpy.tile(marginal_values, (n_grid, 1))
    grid_points[:, other_indices] = marginal_tiled
    Z_vals = self.mcmc_routine.posterior_kde(grid_points.T)
    Z_avg = Z_vals.reshape(n_grid, num_marginal_samples).mean(axis=1).reshape(num_points, num_points)
    return Xi, Xj, Z_avg


## END OF MODULE