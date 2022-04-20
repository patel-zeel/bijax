import stan

coin_code = """
data {
    int<lower=0> n; // number of tosses
    int<lower=0> y; // number of heads
}
transformed data {}
parameters {
    real<lower=0, upper=1> p;
}
transformed parameters {}
model {
    p ~ beta(2, 2);
    y ~ binomial(n, p);
}
generated quantities {}
"""

coin_dat = {
    "n": 100,
    "y": 61,
}

model = stan.build(coin_code, coin_dat, 0)
res = model.vb()

res_df = res.to_frame()
print(res_df["p"].describe())
