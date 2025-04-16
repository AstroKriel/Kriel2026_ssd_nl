## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
import emcee
import corner
from jormi.ww_io import flash_data
from jormi.ww_data import interpolate_data
from jormi.ww_plots import plot_manager


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def log10_model(time, params, init_energy, sat_energy, gamma):
  time_nl_start, time_sat_start, beta = params
  time   = numpy.array(time)
  energy = numpy.zeros_like(time)
  ## mask different ssd phases
  mask_exp_phase = time <= time_nl_start
  mask_nl_phase  = (time_nl_start < time) & (time <= time_sat_start)
  mask_sat_phase = time_sat_start < time
  ## compute energy in different ssd phases
  alpha = (sat_energy - init_energy * numpy.exp(gamma * time_nl_start)) / (time_sat_start - time_nl_start)**beta
  energy[mask_exp_phase] = init_energy * numpy.exp(gamma * time[mask_exp_phase])
  energy[mask_nl_phase]  = init_energy * numpy.exp(gamma * time_nl_start) + alpha * (time[mask_nl_phase] - time_nl_start)**beta
  energy[mask_sat_phase] = init_energy * numpy.exp(gamma * time_nl_start) + alpha * (time_sat_start - time_nl_start)**beta
  return numpy.log10(energy)

def check_params_are_valid(params, max_time, debug_mode=False):
  time_nl_start, time_sat_start, beta = params
  errors = []
  if not (10 < time_nl_start < 0.85 * max_time):
    errors.append(f"`time_nl_start` ({time_nl_start}) must be greater than 0 and less than time_sat_start ({time_sat_start}).")
  if not (time_nl_start < time_sat_start < max_time):
    errors.append(f"`time_sat_start` ({time_sat_start}) must be less than `max_time` = {max_time}.")
  if not (0.25 < beta < 3):
    errors.append(f"`beta` ({beta}) must be between 0 and 10.")
  if len(errors) > 0:
    if debug_mode: print("\n".join(errors))
    return False
  return True

def log_likelihood(params, time, log10_measured_energy, max_time, init_energy, sat_energy, gamma):
  if not check_params_are_valid(params, max_time): return -numpy.inf
  log10_estimated_energy = log10_model(time, params, init_energy, sat_energy, gamma)
  if not numpy.all(numpy.isfinite(log10_estimated_energy)): return -numpy.inf
  return -0.5 * numpy.sum(numpy.square(log10_measured_energy - log10_estimated_energy))

def log_prior(params, max_time):
  if not check_params_are_valid(params, max_time): return -numpy.inf
  return 0 # uniform prior

def log_posterior(params, time, log10_measured_energy, max_time, init_energy, sat_energy, gamma):
  max_time = numpy.max(time)
  log_prior_value = log_prior(params, max_time)
  if not numpy.isfinite(log_prior_value): return -numpy.inf
  log_likelihood_value = log_likelihood(params, time, log10_measured_energy, max_time, init_energy, sat_energy, gamma)
  return log_prior_value + log_likelihood_value

def estimate_params_with_mcmc(time, log10_measured_energy, initial_guess, max_time, init_energy, sat_energy, gamma):
  num_walkers   = 100
  num_steps     = 5000
  burn_in_steps = 1000
  num_params    = len(initial_guess)
  param_positions = numpy.array(initial_guess) + 1e-4 * numpy.random.randn(num_walkers, num_params)
  sampler = emcee.EnsembleSampler(num_walkers, num_params, log_posterior, args=(time, log10_measured_energy, max_time, init_energy, sat_energy, gamma))
  sampler.run_mcmc(param_positions, num_steps)
  samples = sampler.get_chain(discard=burn_in_steps, thin=10, flat=True)
  save_corner_plot(samples)
  plot_chain_evolution(sampler)
  return numpy.median(samples, axis=0)

def save_corner_plot(samples):
  fig = corner.corner(samples)
  plot_manager.save_figure(fig, "mcmc_corner_plot.png")

def plot_chain_evolution(sampler):
  chain = sampler.get_chain()
  _, num_walkers, num_params = chain.shape
  fig, axs = plot_manager.create_figure(num_rows=num_params, axis_shape=(6, 10), share_x=True)
  for param_index in range(num_params):
    ax = axs[param_index]
    for walker_index in range(num_walkers):
      ax.plot(chain[:, walker_index, param_index], alpha=0.3, lw=0.5)
  axs[-1].set_xlabel("steps")
  plot_manager.save_figure(fig, "mcmc_chain_evolution.png")


## ###############################################################
## ESTIMATE TRANSITION
## ###############################################################
def main():
  run_mcmc = 1
  fig, ax = plot_manager.create_figure(axis_shape=(6, 10))
  time_start = 20.0
  time, measured_energy = flash_data.read_vi_data(
    directory    = "/scratch/jh2/nk7952/Re500/Mach0.3/Pm1/576",
    dataset_name = "mag",
    time_start   = time_start,
  )
  time = time - time_start # skip straight to the exponential phase
  log10_measured_energy = numpy.log10(measured_energy) # fit in log10-space
  ## uniformly smaple the data
  num_interp_points = 100
  _interp_time = numpy.linspace(numpy.min(time), numpy.max(time), num_interp_points)
  interp_time, log10_interp_energy = interpolate_data.interpolate_1d(time, log10_measured_energy, _interp_time, kind="linear")
  time = interp_time
  log10_measured_energy = log10_interp_energy
  ## get constraining knowns
  max_time        = numpy.max(time)
  num_points      = len(log10_measured_energy)
  exp_end_index   = int(0.25 * num_points)
  sat_start_index = int(0.75 * num_points)
  init_energy     = 10**(log10_measured_energy[0])
  sat_energy      = 10**numpy.median(log10_measured_energy[sat_start_index:])
  gamma           = numpy.median(numpy.gradient(log10_measured_energy[:exp_end_index], time[:exp_end_index])) / numpy.log10(numpy.exp(1))
  ## sanity check
  my_params = [100, 150, 1.5]
  if not check_params_are_valid(my_params, numpy.max(time), debug_mode=True):
    raise ValueError("Error: my estimated paramaters are invalid!")
  log10_my_estimated_energy = log10_model(time, my_params, init_energy, sat_energy, gamma)
  ## run the mcmc routine
  if run_mcmc:
    init_params = [50, 200, 0.5] # terrible guess
    mcmc_params = estimate_params_with_mcmc(time, log10_measured_energy, init_params, max_time, init_energy, sat_energy, gamma)
    print([
      f"{param:.3f}"
      for param in mcmc_params
    ])
    log10_mcmc_estimated_energy = log10_model(time, mcmc_params, init_energy, sat_energy, gamma)
  ## plot data
  ax.plot(time, log10_measured_energy, color="blue", marker="o", ms=5, ls="", zorder=3, label="measured values")
  ax.plot(time, log10_my_estimated_energy, color="green", marker="o", ms=5, ls="-", lw=2, zorder=3, label="my estimate")
  if run_mcmc: ax.plot(time, log10_mcmc_estimated_energy, color="red", marker="o", ms=5, ls="-", lw=2, zorder=3, label="MCMC estimate")
  ax.axhline(y=numpy.log10(init_energy), color="red", ls="--")
  ax.axhline(y=numpy.log10(sat_energy), color="red", ls="--")
  ax.axvline(x=time[sat_start_index], color="red", ls="--")
  ## label plots
  ax.set_xlabel("time")
  ax.set_ylabel("energy")
  # ax.set_ylabel("$\log_{10}$(energy)")
  ax.legend(loc="lower right")
  plot_manager.save_figure(fig, "estimate_using_mcmc.png")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF SCRIPT