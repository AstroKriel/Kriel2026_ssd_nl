## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from jormi.utils import list_utils
from jormi.ww_io import io_manager
from jormi.ww_plots import plot_manager
from . import mcmc_base


## ###############################################################
## STAGE 2 MCMC FITTER
## ###############################################################
class Stage2MCMC(mcmc_base.BaseMCMCModel):
  def __init__(self, output_directory, time, measured, stage1_params, verbose):
    log10_init_energy, self.transition_time, self.gamma = stage1_params
    self.init_energy       = 10**(log10_init_energy)
    log10_sat_energy_guess = numpy.log10(self.init_energy * numpy.exp(self.gamma * self.transition_time))
    transition_index       = list_utils.get_index_of_closest_value(time, self.transition_time)
    super().__init__(
      output_directory = output_directory,
      routine_name = "stage2",
      verbose      = verbose,
      time         = time,
      measured     = measured,
      ll_sigma     = numpy.std(measured[transition_index:]),
      param_guess  = (0.85 * self.transition_time, 1.25 * self.transition_time, log10_sat_energy_guess),
      param_labels = [
        r"$t_{\mathrm{nl}}$",
        r"$t_{\mathrm{sat}}$",
        r"$\log_{10}(E_{\mathrm{sat}})$"
      ]
    )

  def _model(self, fit_params):
    (start_nl_time, start_sat_time, log10_sat_energy) = fit_params
    ## mask different ssd phases
    mask_exp_phase = self.time < start_nl_time
    mask_nl_phase  = (start_nl_time <= self.time) & (self.time < start_sat_time)
    mask_sat_phase = start_sat_time < self.time
    ## calculate model constants
    start_nl_energy = self.init_energy * numpy.exp(self.gamma * start_nl_time)
    sat_energy      = 10**log10_sat_energy
    alpha           = (sat_energy - start_nl_energy) / (start_sat_time - start_nl_time)
    ## model energy evolution
    energy = numpy.zeros_like(self.time)
    energy[mask_exp_phase] = self.init_energy * numpy.exp(self.gamma * self.time[mask_exp_phase])
    energy[mask_nl_phase]  = start_nl_energy + alpha * (self.time[mask_nl_phase] - start_nl_time)
    energy[mask_sat_phase] = sat_energy
    return energy

  def _check_params_are_valid(self, fit_params, print_errors=False):
    (start_nl_time, start_sat_time, log10_sat_energy) = fit_params
    errors = []
    if not (0.1 * self.max_time < start_nl_time < self.transition_time):
      errors.append(f"`start_nl_time` ({start_nl_time:.2f}) must be larger than 0.1 * `max_time` ({self.max_time:.2f}) and smaller than the stage-1 estimated transition time ({self.transition_time:.2f}).")
    if not (self.transition_time < start_sat_time < self.max_time):
      errors.append(f"`start_sat_time` ({start_sat_time:.2f}) must be larger than the stage-1 estimated transition time ({self.transition_time:.2f}) and less than `max_time` ({self.max_time:.2f}).")
    if not (-5 < log10_sat_energy < 0):
      errors.append(f"`log10_sat_energy` ({log10_sat_energy:.2f}) must be between -5 and 0.")
    if len(errors) > 0:
      if print_errors: print("\n".join(errors))
      return False
    return True

  def _plot_model_results(self, fit_params):
    (start_nl_time, start_sat_time, log10_sat_energy) = fit_params
    fig, axs  = plot_manager.create_figure(num_rows=3, share_x=True)
    data_args = dict(color="blue", marker="o", ms=5, ls="-", lw=1.0, zorder=3)
    measured_dy_dt = numpy.gradient(self.measured, self.time)
    axs[0].plot(self.time, self.measured, **data_args)
    axs[1].plot(self.time, measured_dy_dt, **data_args)
    model_args      = dict(color="red", ls="-", lw=1.5, zorder=5)
    start_nl_energy = self.init_energy * numpy.exp(self.gamma * start_nl_time)
    sat_energy      = 10**log10_sat_energy
    alpha           = (sat_energy - start_nl_energy) / (start_sat_time - start_nl_time)
    modelled        = self._model(fit_params)
    residual        = (self.measured - modelled) / self.ll_sigma
    axs[0].plot(self.time, modelled, **model_args)
    axs[1].axhline(y=alpha, color="red", ls="--", lw=1.5)
    axs[2].plot(self.time, residual, **model_args)
    for row_index in range(len(axs)):
      ax = axs[row_index]
      ax.axvline(x=start_nl_time, color="red", ls="--", lw=1.5)
      ax.axvline(x=start_sat_time, color="red", ls="--", lw=1.5)
      ax.axhline(y=0.0, color="black", ls="--")
    axs[0].set_ylabel(r"$E_{\rm mag}$")
    axs[1].set_ylabel(r"$({\rm d}/{\rm d}t) E_{\rm mag}$")
    axs[2].set_ylabel(r"residuals")
    axs[-1].set_xlabel("t")
    fig_name = f"{self.routine_name}_fit.png"
    fig_file_path = io_manager.combine_file_path_parts([ self.output_directory, fig_name ])
    plot_manager.save_figure(fig, fig_file_path, verbose=self.verbose)


## END OF MODULE