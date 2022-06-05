# tabularasa

## Tabular Pytorch Neural Networks with monotonicity, uncertainty, and a scikit API

### Overview

This library is heavily indebted to:

- TODO: UMNN paper and repo 
- TODO: Orthonormal certificates paper and repo

With the goal to provide a usable open source implementation with minimal overhead and more flexibility.

### Usage

```
import tabularasa as tr

model = tr.RasaRegressor()
```

#### How to define your own network

TODO: What does `.forward()` need to include?

### FAQ

**Why is the package named "tabularasa"?**

- I'm not a strong proponent of the tabula rasa theory of development, I just wanted a name with "tabular" in it.  Plus, I like that tabula rasa hints at the ability to learn anything, which ideally (although not practically) this package could.

**What is the long-term plan for tabularasa?**

- Ideally (in order of how much I care about them):
  - Continue to improve and evolve the default network that's used when a specific network isn't specified.
  - Better software engineering practices (tests, error messages, etc.).
  - Potentially expand into problems beyond regression.  It isn't necessarily clear how to generalize monotonicity for multiclass classification problems, but I'd be interested if others see value here.
