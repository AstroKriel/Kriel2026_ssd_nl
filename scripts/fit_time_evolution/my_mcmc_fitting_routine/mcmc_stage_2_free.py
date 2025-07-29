## ###############################################################
## DEPENDENCIES
## ###############################################################

import numpy
from scipy.ndimage import gaussian_filter1d
from jormi.utils import list_utils
from . import base_mcmc
from . import mcmc_utils


## ###############################################################
## STAGE 2 MCMC FITTER
## ###############################################################

class Stage2MCMCRoutine_free(base_mcmc.BaseMCMCRoutine):
  def __init__(
      self,
      *,
      output_directory   : str,
      time_values        : list | numpy.ndarray,
      ave_energy_values  : list | numpy.ndarray,
      std_energy_values  : list | numpy.ndarray,
      initial_params     : tuple[float, ...],
      is_supersonic      : bool,
      prior_kde          : callable = None,
      plot_posterior_kde : bool = True,
    ):
    guess_t_sat = self._define_constraints(time_values, ave_energy_values, is_supersonic)
    super().__init__(
      routine_name        = "stage2_free",
      output_directory    = output_directory,
      x_values            = time_values,
      y_values            = ave_energy_values,
      likelihood_sigma    = std_energy_values,
      initial_params      = (
        initial_params[0], # log10(E_init)
        initial_params[1], # log10(E_sat)
        initial_params[2], # gamma
        initial_params[3], # t_nl
        guess_t_sat, # t_sat
        initial_params[5] # exponent
      ),
      prior_kde           = prior_kde,
      plot_posterior_kde  = plot_posterior_kde,
      data_label          = r"$E_{\mathrm{mag}}$",
      fitted_param_labels = [
        r"$\log_{10}(E_{\mathrm{init}})$",
        r"$\log_{10}(E_{\mathrm{sat}})$",
        r"$\gamma$",
        r"$t_{\mathrm{nl}}$",
        r"$t_{\mathrm{sat}}$",
        r"$p$"
      ]
    )

  def _define_constraints(self, time_values, ave_energy_values, is_supersonic):
    dy_dt = numpy.gradient(ave_energy_values, time_values)
    dlny_dt = gaussian_filter1d(numpy.gradient(numpy.log10(ave_energy_values), time_values), sigma=2)
    if is_supersonic: target_dy_dt = 0.25 * numpy.max(dy_dt)
    else: target_dy_dt = 0.5 * numpy.max(dy_dt)
    max_t_nl_index = list_utils.find_first_crossing(values=dy_dt, target=target_dy_dt)
    max_t_sat_index = list_utils.find_first_crossing(values=dlny_dt, target=0)
    max_t_nl = time_values[max_t_nl_index]
    max_t_sat = time_values[max_t_sat_index]
    guess_t_sat = max_t_nl + 0.5 * (max_t_sat - max_t_nl)
    return guess_t_sat

  def _model(self, param_vectors):
    param_vectors = numpy.atleast_2d(param_vectors) # (N, P)
    ## output dimensions
    num_local_walkers = param_vectors.shape[0] # N
    num_data_points = len(self.x_values) # T
    ## unpack model parameters (P = 6)
    log10_init_energy, log10_sat_energy, gamma, start_nl_time, start_sat_time, exponent = param_vectors.T
    ## reshape parameters to allow for vectorising over param-rows
    x_values_2d        = self.x_values[None, :] # shape (1, T)
    gamma_2d           = gamma[:, None] # shape (N, 1)
    start_nl_time_2d   = start_nl_time[:, None] # shape (N, 1)
    start_sat_time_2d  = start_sat_time[:, None] # shape (N, 1)
    exponent_2d        = exponent[:, None] # shape (N, 1)
    ## mask SSD phases
    mask_exp_phase     = x_values_2d < start_nl_time_2d
    mask_nl_phase      = (start_nl_time_2d <= x_values_2d) & (x_values_2d < start_sat_time_2d)
    mask_sat_phase     = start_sat_time_2d < x_values_2d
    ## compute model constants (per walker)
    init_energy        = 10**log10_init_energy # (N,)
    init_energy_2d     = init_energy[:, None] # (N, 1)
    sat_energy         = 10**log10_sat_energy # (N,)
    sat_energy_2d      = sat_energy[:, None] # (N, 1)
    start_nl_energy    = init_energy * numpy.exp(gamma * start_nl_time) # (N,)
    start_nl_energy_2d = start_nl_energy[:, None] # (N, 1)
    alpha              = (sat_energy - start_nl_energy) / (start_sat_time - start_nl_time)**exponent # (N,)
    alpha_2d           = alpha[:, None] # (N, 1)
    ## assemble modelled SSD phases
    energy_2d = numpy.zeros((num_local_walkers, num_data_points))
    energy_2d[mask_exp_phase] = (init_energy_2d * numpy.exp(gamma_2d * x_values_2d))[mask_exp_phase] # (N, T)
    energy_2d[mask_nl_phase]  = (start_nl_energy_2d + alpha_2d * (x_values_2d - start_nl_time_2d)**exponent_2d)[mask_nl_phase] # (N, T)
    energy_2d[mask_sat_phase] = numpy.broadcast_to(sat_energy_2d, (num_local_walkers, num_data_points))[mask_sat_phase] # (N, T)
    return energy_2d

  def _get_valid_params_mask(self, param_vectors):
    param_vectors = numpy.atleast_2d(param_vectors)
    num_local_walkers = param_vectors.shape[0]
    log10_init_energy, log10_sat_energy, gamma, start_nl_time, start_sat_time, exponent = param_vectors.T
    valid_log10_init_energy = (-30 < log10_init_energy) & (log10_init_energy < -5)
    valid_log10_sat_energy  = (-5 < log10_sat_energy) & (log10_sat_energy < 0)
    valid_gamma             = (0 < gamma) & (gamma < 10)
    valid_start_nl_time     = (0.1 * self.max_time < start_nl_time) & (start_nl_time < self.max_t_nl) & (start_nl_time < start_sat_time)
    valid_start_sat_time    = start_sat_time < self.max_t_sat
    valid_exponent          = (1.0 < exponent) & (exponent < 2.0)
    valid_params_mask = (
      valid_log10_init_energy &
      valid_log10_sat_energy &
      valid_gamma &
      valid_start_nl_time &
      valid_start_sat_time &
      valid_exponent
    )
    if num_local_walkers == 1:
      return valid_params_mask[0]
    return valid_params_mask

  def _get_kde_params(self, param_vectors):
    ## ignore the transition times implicity gives them a unifrom prior
    return numpy.asarray(param_vectors[:, :3])

  def _annotate_fitted_params(self, axs):
    sat_energy_samples      = 10**self.fitted_posterior_samples[:,1]
    start_nl_time_samples   = self.fitted_posterior_samples[:,3]
    start_sat_time_samples  = self.fitted_posterior_samples[:,4]
    mcmc_utils.plot_param_percentiles(axs[0], sat_energy_samples, orientation="horizontal")
    for row_index in range(len(axs)):
      mcmc_utils.plot_param_percentiles(axs[row_index], start_nl_time_samples, orientation="vertical")
      mcmc_utils.plot_param_percentiles(axs[row_index], start_sat_time_samples, orientation="vertical")
      axs[row_index].axvline(self.max_t_nl, color="red")
      axs[row_index].axvline(self.max_t_sat, color="red")


## END OF MODULE