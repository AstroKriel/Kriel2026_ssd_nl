## ###############################################################
## DEPENDENCIES
## ###############################################################

import numpy
from . import base_mcmc


## ###############################################################
## HELPER FUNCTION
## ###############################################################

def plot_param_percentiles(ax, samples, orientation):
  p16, p50, p84 = numpy.percentile(samples, [16, 50, 84])
  if   "h" in orientation.lower():
    ax_line = ax.axhline
    ax_span = ax.axhspan
  elif "v" in orientation.lower():
    ax_line = ax.axvline
    ax_span = ax.axvspan
  else: raise ValueError("`orientation` must either be `horizontal` (`h`) or `vertical` (`v`).")
  ax_line(p50, color="green", ls=":", lw=1.5, zorder=5)
  ax_span(p16, p84, color="green", ls="-", lw=1.5, alpha=0.3, zorder=4)


## ###############################################################
## STAGE 2 MCMC FITTER
## ###############################################################

class MCMCStage2Routine(base_mcmc.BaseMCMCRoutine):
  def __init__(
      self,
      *,
      output_directory : str,
      x_values         : list | numpy.ndarray,
      y_values         : list | numpy.ndarray,
      initial_params   : tuple[float, ...],
      likelihood_sigma : float = 1.0,
      prior_kde        = None,
      verbose          : bool = True,
      debug_mode       : bool = False
    ):
    self.max_time = numpy.max(x_values)
    super().__init__(
      output_directory    = output_directory,
      routine_name        = "stage2",
      verbose             = verbose,
      debug_mode          = debug_mode,
      x_values            = x_values,
      y_values            = y_values,
      prior_kde           = prior_kde,
      likelihood_sigma    = likelihood_sigma,
      initial_params      = initial_params,
      y_data_label        = r"$E_{\mathrm{mag}}$",
      fitted_param_labels = [
        r"$\log_{10}(E_{\mathrm{init}})$",
        r"$\log_{10}(E_{\mathrm{sat}})$",
        r"$\gamma$",
        r"$t_{\mathrm{nl}}$",
        r"$t_{\mathrm{sat}}$",
      ]
    )

  def _model(self, fit_params):
    (log10_init_energy, log10_sat_energy, gamma, start_nl_time, start_sat_time) = fit_params
    ## mask different ssd phases
    mask_exp_phase = self.x_values < start_nl_time
    mask_nl_phase  = (start_nl_time <= self.x_values) & (self.x_values < start_sat_time)
    mask_sat_phase = start_sat_time < self.x_values
    ## compute model constants
    init_energy     = 10**log10_init_energy
    sat_energy      = 10**log10_sat_energy
    start_nl_energy = init_energy * numpy.exp(gamma * start_nl_time)
    alpha           = (sat_energy - start_nl_energy) / (start_sat_time - start_nl_time)
    ## model energy evolution
    energy = numpy.zeros_like(self.x_values)
    energy[mask_exp_phase] = init_energy * numpy.exp(gamma * self.x_values[mask_exp_phase])
    energy[mask_nl_phase]  = start_nl_energy + alpha * (self.x_values[mask_nl_phase] - start_nl_time)
    energy[mask_sat_phase] = sat_energy
    return energy

  def _check_params_are_valid(self, fit_params, print_errors=False):
    (log10_init_energy, log10_sat_energy, gamma, start_nl_time, start_sat_time) = fit_params
    errors = []
    if not (-30 < log10_init_energy < -5):
      errors.append(f"`log10_init_energy` ({log10_init_energy:.2f}) must be between -20 and -5.")
    if not (-5 < log10_sat_energy < 0):
      errors.append(f"`log10_sat_energy` ({log10_sat_energy:.2f}) must be between -5 and 0.")
    if not (0 < gamma < 2):
      errors.append(f"`gamma` ({gamma:.2f}) must be between 0 and 2.")
    if not (0.1 * self.max_time < start_nl_time < start_sat_time):
      errors.append(f"`start_nl_time` ({start_nl_time:.2f}) must be larger than 10% of `max_time` ({self.max_time:.2f}) and smaller than `start_sat_time` ({start_sat_time:.2f}).")
    if not (start_sat_time < self.max_time):
      errors.append(f"`start_sat_time` ({start_sat_time:.2f}) must be smaller than `max_time` ({self.max_time:.2f}).")
    if len(errors) > 0:
      if print_errors: print("\n".join(errors))
      return False
    return True

  def _get_kde_eval_params(self, param_vector: tuple[float, ...]) -> numpy.ndarray:
    ## transition times have unifrom prior
    return numpy.asarray(param_vector[:3])

  def _annotate_fitted_params(self, axs):
    gamma_samples             = self.fitted_posterior_samples[:,2]
    start_nl_time_samples     = self.fitted_posterior_samples[:,3]
    start_sat_time_samples    = self.fitted_posterior_samples[:,4]
    init_energy_samples       = 10**self.fitted_posterior_samples[:,0]
    sat_energy_samples        = 10**self.fitted_posterior_samples[:,1]
    start_nl_energy_samples   = init_energy_samples * numpy.exp(gamma_samples * start_nl_time_samples)
    alpha_samples             = (sat_energy_samples - start_nl_energy_samples) / (start_sat_time_samples - start_nl_time_samples)
    plot_param_percentiles(axs[0], sat_energy_samples, orientation="horizontal")
    plot_param_percentiles(axs[1], alpha_samples, orientation="horizontal")
    for row_index in range(len(axs)):
      plot_param_percentiles(axs[row_index], start_nl_time_samples, orientation="vertical")
      plot_param_percentiles(axs[row_index], start_sat_time_samples, orientation="vertical")


## END OF MODULE