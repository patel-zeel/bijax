## AJAX

This repo contains several approximate Bayesian inference algorithms implemented in JAX.

### Installation

```
pip install git+https://github.com/patel-zeel/ajax.git
```

### Basic usage
```py
from ajax.advi import ADVI
from ajax.laplace import ADLaplace
from ajax.mcmc import NUTS  # TBD
```

### Core Principals

* Each component in prior should return a single scalar `log_prob`.
