from cmdstanpy import cmdstan_path, CmdStanModel

model = CmdStanModel(stan_file="lab/cmdstanpy/coin_toss.stan")
print(model)
