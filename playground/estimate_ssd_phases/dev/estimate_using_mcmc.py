## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
import emcee
import corner
import matplotlib.pyplot as mpl_plot
from loki.ww_plots import plot_manager
from loki.ww_io import flash_data
import utils # local


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def model(time, init_energy, gamma, transition_time):
  return numpy.where(
    time < transition_time,
    init_energy + gamma * time,
    init_energy + gamma * transition_time
  )

def check_params_are_valid(params):
  init_energy, gamma, transition_time = params
  if not (-100 < init_energy < 0): return False
  if not (0 < gamma < 1): return False
  if transition_time <= 0 or 1e3 <= transition_time: return False
  return True

def log_likelihood(params, time, measured_energy):
  if not check_params_are_valid(params): return -numpy.inf
  modelled_energy = model(time, *params)
  return -0.5 * numpy.sum(numpy.square(measured_energy - modelled_energy))

def log_prior(params):
  if not check_params_are_valid(params): return -numpy.inf
  return 0 # uniform prior

def log_posterior(params, time, measured_energy):
  log_prior_value      = log_prior(params)
  if not numpy.isfinite(log_prior_value): return -numpy.inf
  log_likelihood_value = log_likelihood(params, time, measured_energy)
  return log_prior_value + log_likelihood_value

def estimate_params(time, measured_energy, params_truth):
  num_walkers     = 100
  num_steps       = 5000
  burn_in_steps   = 1000
  initial_guess   = [-10, 1e-2, 100]
  num_params      = len(initial_guess)
  param_positions = initial_guess + 1e-4 * numpy.random.randn(num_walkers, num_params)
  sampler = emcee.EnsembleSampler(num_walkers, num_params, log_posterior, args=(time, measured_energy))
  sampler.run_mcmc(param_positions, num_steps)
  samples = sampler.get_chain(discard=burn_in_steps, thin=10, flat=True)
  fig_corner = corner.corner(samples, truths=params_truth)
  fig_corner.savefig("mcmc_corner_plot.png")
  mpl_plot.close(fig_corner)
  init_energy, growth_rate, transition_time = numpy.median(samples, axis=0)
  return [ init_energy, growth_rate, transition_time ]

def generate_data(num_points, time_bounds, init_energy, growth_rate, transition_time):
  time = utils.generate_uniform_domain(
    domain_bounds = time_bounds,
    num_points    = num_points,
  )
  measured_energy = utils.generate_data(
    x_data       = time,
    noise_level  = 3.0,
    init_value   = init_energy,
    growth_rate  = growth_rate,
    x_transition = transition_time,
  )
  return time, measured_energy

def load_data():
  time_start = 20.0
  time, measured_energy = flash_data.read_vi_data(
    directory    = "/scratch/jh2/nk7952/Re1500/Mach0.1/Pm1/144",
    dataset_name = "mag",
    time_start   = time_start,
  )
  time = time - time_start
  measured_energy = numpy.log10(measured_energy)
  return time, measured_energy


## ###############################################################
## ESTIMATE TRANSITION
## ###############################################################
def main():
  fig, ax = plot_manager.create_figure(axis_shape=(6, 10))
  init_energy      = -14
  growth_rate     = 0.05
  transition_time = 220
  # time, measured_energy = generate_data(100, [0, 500], init_energy, growth_rate, transition_time)
  time, measured_energy = load_data()
  true_params = [ init_energy, growth_rate, transition_time ]
  estimated_params = estimate_params(time, measured_energy, true_params)
  [
    print(f"{param:.3f}")
    for param in estimated_params
  ]
  # estimated_params = [-14, 0.05, 220]
  modelled_energy  = model(time, *estimated_params)
  ax.plot(time, measured_energy, color="blue", marker="o", ms=5, ls="", zorder=3, label="measured values")
  ax.plot(time, modelled_energy, color="red", marker="o", ms=5, ls="-", lw=2, zorder=3, label="MCMC estimated model")
  ax.set_xlabel("time")
  ax.set_ylabel("energy")
  plot_manager.save_figure(fig, "estimate_using_mcmc.png")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF SCRIPT