## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from jormi.ww_io import io_manager
from jormi.ww_plots import plot_manager
from . import mcmc_base


## ###############################################################
## STAGE 1 MCMC FITTER
## ###############################################################
class Stage1MCMC(mcmc_base.BaseMCMCModel):
  def __init__(self, output_directory, x_data, y_data, verbose):
    self.log10_e = numpy.log10(numpy.exp(1))
    self.max_time = numpy.max(x_data)
    super().__init__(
      output_directory = output_directory,
      routine_name     = "stage1",
      verbose          = verbose,
      x_data           = x_data,
      y_data           = numpy.log10(y_data),
      param_guess      = (-20, 0.85 * numpy.max(x_data), 0.5),
      param_labels     = [
        r"$\log_{10}(E_{\mathrm{init}})$",
        r"$t_{\mathrm{approx}}$",
        r"$\gamma$"
      ]
    )

  def _model(self, fit_params):
    (log10_init_energy, transition_time, gamma) = fit_params
    ## mask into two rough phases
    mask_exp = self.x_data < transition_time
    mask_sat = ~mask_exp
    ## model log_10 energy evolution
    log10_energy = numpy.zeros_like(self.x_data)
    log10_energy[mask_exp] = log10_init_energy + self.log10_e * gamma * self.x_data[mask_exp]
    log10_energy[mask_sat] = log10_init_energy + self.log10_e * gamma * transition_time
    return log10_energy

  def _check_params_are_valid(self, fit_params, print_errors=False):
    (log10_init_energy, transition_time, gamma) = fit_params
    errors = []
    if not (-30 < log10_init_energy < -5):
      errors.append(f"`log10_init_energy` ({log10_init_energy:.2f}) must be between -20 and -5.")
    if not (0.25 * self.max_time < transition_time < 0.9 * self.max_time):
      errors.append(f"`transition_time` ({transition_time:.2f}) must be between 25 and 90 percent of `max_time` ({self.max_time:.2f}).")
    if not (0 < gamma < 1):
      errors.append(f"`gamma` ({gamma:.2f}) must be between 0 and 1.")
    if len(errors) > 0:
      if print_errors: print("\n".join(errors))
      return False
    return True

  def _plot_model_results(self, fit_params):
    (_, transition_time, gamma) = fit_params
    fig, axs = plot_manager.create_figure(num_rows=3, share_x=True)
    data_args = dict(color="blue", marker="o", ms=5, ls="-", lw=1.0, zorder=3)
    measured_dlog10y_dt = numpy.gradient(self.y_data, self.x_data)
    axs[0].plot(self.x_data, self.y_data, **data_args)
    axs[1].plot(self.x_data, measured_dlog10y_dt, **data_args)
    model_args = dict(color="red", ls="-", lw=1.5, zorder=5)
    modelled_y = self._model(fit_params)
    residuals  = self.y_data - modelled_y
    axs[0].plot(self.x_data, modelled_y, **model_args)
    axs[1].axhline(y=self.log10_e * gamma, color="red", ls="--", lw=1.5)
    axs[2].plot(self.x_data, residuals, **model_args)
    for row_index in range(len(axs)):
      axs[row_index].axvline(x=transition_time, color="red", ls="--", lw=1.5)
    axs[1].axhline(y=0.0, color="black", ls="--")
    axs[2].axhline(y=0.0, color="black", ls="--")
    axs[0].set_ylabel(r"$\log_{10}(E_{\rm mag})$")
    axs[1].set_ylabel(r"$({\rm d}/{\rm d}t) \log_{10}(E_{\rm mag})$")
    axs[2].set_ylabel(r"residuals")
    axs[2].set_xlabel("t")
    fig_name = f"{self.routine_name}_fit.png"
    fig_file_path = io_manager.combine_file_path_parts([ self.output_directory, fig_name ])
    plot_manager.save_figure(fig, fig_file_path, verbose=self.verbose)


## END OF MODULE