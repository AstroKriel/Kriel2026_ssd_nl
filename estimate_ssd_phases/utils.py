import numpy

def generate_uniform_domain(
    domain_bounds : tuple[float, float],
    num_points    : int,
  ) -> numpy.ndarray:
  x_data = numpy.linspace(domain_bounds[0], domain_bounds[1], 100, num_points)
  return x_data

def generate_nonuniform_domain(
    domain_bounds : tuple[float, float],
    num_points    : int,
) -> numpy.ndarray:
  x_values = generate_uniform_domain(
    domain_bounds = domain_bounds,
    num_points    = num_points,
  )
  numpy.random.seed(0)
  max_perturb = (domain_bounds[1] - domain_bounds[0]) / 10
  x_perturbs  = numpy.random.uniform(-max_perturb, max_perturb, size=num_points)
  x_perturbed_values = numpy.sort(x_values + x_perturbs)
  x_min, x_max = x_perturbed_values[0], x_perturbed_values[-1]
  x_perturbed_values = 101 * (x_perturbed_values - x_min) / (x_max - x_min)
  return x_perturbed_values

def generate_data(
    x_data       : numpy.ndarray,
    noise_level  : float,
    init_value   : float = 1.0,
    growth_rate  : float = 10.0,
    x_transition : float = 50.0,
  ):
  y_data = numpy.piecewise(
    x_data,
    [
      x_data < x_transition,
      x_data >= x_transition
    ],
    [
      lambda x: init_value + growth_rate * x,
      init_value + growth_rate * x_transition
    ]
  )
  numpy.random.seed(42)
  y_data += 0.5 * growth_rate * numpy.random.normal(0, noise_level, x_data.shape)
  return y_data

## END OF UTILITY MODULE