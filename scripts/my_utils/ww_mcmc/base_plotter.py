## ###############################################################
## BASE ROUTINE
## ###############################################################

class BaseMCMCPlotter:
  def __init__(self, mcmc_routine):
    self.mcmc_routine = mcmc_routine

  def plot(self):
    raise NotImplementedError()


## END OF MODULE