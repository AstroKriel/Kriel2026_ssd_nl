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
    threshold = 0.05 * numpy.max(pdf)
    i1 = list_utils.find_first_crossing(values=pdf, target=threshold, direction="rising")
    # i2 = list_utils.find_first_crossing(values=pdf, target=threshold, direction="falling")
    reversed_pdf = pdf[::-1]
    i2_rev = list_utils.find_first_crossing(values=reversed_pdf, target=threshold, direction="falling")
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


## END OF MODULE