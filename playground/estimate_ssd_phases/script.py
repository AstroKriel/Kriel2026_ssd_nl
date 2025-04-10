## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
import emcee
import corner
import matplotlib.pyplot as mpl_plot
from loki.ww_plots import plot_manager
from loki.ww_io import flash_data


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def model(time, log10_init_energy, gamma_exp, time_nl, exponent_nl, time_sat, log10_sat_energy):
  time   = numpy.array(time)
  energy = numpy.zeros_like(time)
  init_energy = 10**log10_init_energy
  sat_energy  = 10**log10_sat_energy
  mask_exp = time <= time_nl
  mask_nl  = (time_nl < time) & (time <= time_sat)
  mask_sat = time_sat < time
  alpha_nl = (sat_energy - init_energy * numpy.exp(gamma_exp * time_nl)) / numpy.power(time_sat - time_nl, exponent_nl)
  energy[mask_exp] = init_energy * numpy.exp(gamma_exp * time[mask_exp])
  energy[mask_nl]  = init_energy * numpy.exp(gamma_exp * time_nl) + alpha_nl * (time[mask_nl] - time_nl)**exponent_nl
  energy[mask_sat] = init_energy * numpy.exp(gamma_exp * time_nl) + alpha_nl * (time_sat - time_nl)**exponent_nl
  return energy

def check_params_are_valid(params, max_time, max_log10_energy, debug_mode=False):
  log10_init_energy, gamma_exp, time_nl, exponent_nl, time_sat, log10_sat_energy = params
  errors = []
  if not (-20 < log10_init_energy < -10):
    errors.append(f"`log10_init_energy` ({log10_init_energy}) must be less than `max_log10_energy` = {max_log10_energy}.")
  if not (0 < gamma_exp < 2.0):
    errors.append(f"`gamma_exp` ({gamma_exp}) must be between 0 and 2.")
  if not (100 < time_nl < time_sat):
    errors.append(f"`time_nl` ({time_nl}) must be greater than 0 and less than time_sat ({time_sat}).")
  if not (200 < time_sat < max_time):
    errors.append(f"`time_sat` ({time_sat}) must be less than `max_time` = {max_time}.")
  if not (0 < exponent_nl < 10):
    errors.append(f"`exponent_nl` ({exponent_nl}) must be between 0 and 10.")
  if not (-10 < log10_sat_energy < 1):
    errors.append(f"`log10_sat_energy` ({log10_sat_energy}) must be between -10 and 1.")
  if not (gamma_exp * time_nl < 100):
    errors.append(f"exponential growth factor is too large: `{gamma_exp * time_nl:.3f}` > 100")
  if len(errors) > 0:
    if debug_mode: print("\n".join(errors))
    return False
  return True

def log_likelihood(params, time, measured_energy, max_time, max_log10_energy):
  if not check_params_are_valid(params, max_time, max_log10_energy): return -numpy.inf
  modelled_energy = model(time, *params)
  if not numpy.all(numpy.isfinite(modelled_energy)): return -numpy.inf
  return -0.5 * numpy.sum(numpy.square(measured_energy - modelled_energy))

def log_prior(params, max_time, max_log10_energy):
  if not check_params_are_valid(params, max_time, max_log10_energy): return -numpy.inf
  return 0 # uniform prior

def log_posterior(params, time, measured_energy):
  max_time = numpy.max(time)
  max_log10_energy = numpy.max(numpy.log10(measured_energy))
  log_prior_value = log_prior(params, max_time, max_log10_energy)
  if not numpy.isfinite(log_prior_value): return -numpy.inf
  log_likelihood_value = log_likelihood(params, time, measured_energy, max_time, max_log10_energy)
  return log_prior_value + log_likelihood_value

def estimate_params_with_mcmc(time, measured_energy, initial_guess):
  num_walkers     = 100
  num_steps       = 5000
  burn_in_steps   = 1000
  num_params      = len(initial_guess)
  param_positions = numpy.array(initial_guess) + 1e-4 * numpy.random.randn(num_walkers, num_params)
  sampler = emcee.EnsembleSampler(num_walkers, num_params, log_posterior, args=(time, measured_energy))
  sampler.run_mcmc(param_positions, num_steps)
  samples = sampler.get_chain(discard=burn_in_steps, thin=10, flat=True)
  fig_corner = corner.corner(samples)
  fig_name = "mcmc_corner_plot.png"
  fig_corner.savefig(fig_name)
  mpl_plot.close(fig_corner)
  print(f"Saved figure: {fig_name}")
  return numpy.median(samples, axis=0)


## ###############################################################
## ESTIMATE TRANSITION
## ###############################################################
def main():
  log10_transform = True
  fig, ax = plot_manager.create_figure(axis_shape=(6, 10))
  time_start = 20.0
  time, measured_energy = flash_data.read_vi_data(
    directory    = "/scratch/jh2/nk7952/Re1500/Mach0.1/Pm1/144",
    dataset_name = "mag",
    time_start   = time_start,
  )
  time = time - time_start
  max_time = numpy.max(time)
  max_log10_energy = numpy.max(numpy.log10(measured_energy))
  ## sanity check
  my_params = [-14, 0.11, 160, 3.0, 250, -3.1]
  if not check_params_are_valid(my_params, max_time, max_log10_energy, debug_mode=True):
    raise ValueError("Error: my estimated paramaters are invalid!")
  my_modelled_energy = model(time, *my_params)
  ## emply mcmc routine
  mcmc_params = estimate_params_with_mcmc(time, measured_energy, my_params)
  print(mcmc_params)
  mcmc_modelled_energy = model(time, *mcmc_params)
  if log10_transform:
    measured_energy      = numpy.log10(measured_energy)
    my_modelled_energy   = numpy.log10(my_modelled_energy)
    mcmc_modelled_energy = numpy.log10(mcmc_modelled_energy)
  ax.plot(time, measured_energy, color="blue", marker="o", ms=5, ls="", zorder=3, label="measured values")
  ax.plot(time, my_modelled_energy, color="green", marker="o", ms=5, ls="-", lw=2, zorder=3, label="my estimated model")
  ax.plot(time, mcmc_modelled_energy, color="red", marker="o", ms=5, ls="-", lw=2, zorder=3, label="MCMC estimated model")
  ## label plots
  ax.set_xlabel("time")
  if log10_transform: ax.set_ylabel("log$_{10}$(energy)")
  else: ax.set_ylabel("energy")
  ax.legend(loc="lower right")
  plot_manager.save_figure(fig, "estimate_using_mcmc.png")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF SCRIPT