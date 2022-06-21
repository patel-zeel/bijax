## AJAX

This repo contains several approximate Bayesian inference algorithms implemented in JAX. The original paper is [here](https://www.jmlr.org/papers/volume18/16-107/16-107.pdf).

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
