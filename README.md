# tabularasa

## Tabular Pytorch Neural Networks with monotonicity, uncertainty, and a scikit API

### Overview

This library is heavily indebted to the following works:

- [Antoine Wehenkel and Gilles Louppe. "Unconstrained Monotonic Neural Networks." (2019)](https://arxiv.org/abs/1908.05164) ([repo](https://github.com/AWehenkel/generalized-UMNN/)).
- [Natasa Tagasovska and David Lopez-Paz "Single-Model Uncertainties for Deep Learning" (2019)](https://arxiv.org/abs/1811.00908) ([repo](https://github.com/facebookresearch/SingleModelUncertainty)).
- [Huang et al. "TabTransformer: Tabular Data Modeling Using Contextual Embeddings" (2020)](https://arxiv.org/abs/2012.06678) ([repo](https://github.com/lucidrains/tab-transformer-pytorch))

With the goal to provide a usable open source implementation that combines the functionality of all three papers with minimal overhead and more flexibility.

### Usage

Please see the example notebooks for a walkthrough of how to use TabulaRasa:

1. [example_data](./examples/example_data.ipynb): Generates a fake dataset used throughout the remaining examples.
1. [simple_mlp](./examples/simple_mlp.ipynb): Train a simple multi-layer perceptron with an embedding for the categorical feature, linear layers, and ReLU activation.  Its purpose is to illustrate the skorch API (for those that are unfamiliar), and to show that without constraints, one feature's relationship with the target will be non-monotonic.
1. [mixed_monotnic](./examples/mixed_monotonic.ipynb): Trains a similar network to the simple MLP, but with a monotonic constraint on some features.  In addition, this notebook illustrates the use of orthonormal certificates to estimate epistemic uncertainty based on the training data provided.
1. [simultaneous_quantiles](./examples/simultaneous_quantiles.ipynb): Trains a network similar to the simple MLP, but uses a loss function that can generate estimates for any predicted quantile.  This model does not constrain features to have a monotonic relationship with the target.  These predicted quantiles can be used as estimates for aleatoric uncertainty.
1. [external_monotonic](./examples/external_monotonic.ipynb): Trains a network with a monotonic constraint on some features, however instead of a simple embedding network to handle categorical features, a network from an external package, [TabTransformer](https://github.com/lucidrains/tab-transformer-pytorch), is used.
1. [tabula_rasa](./examples/tabula_rasa.ipynb): Trains a `TabulaRasaRegressor()`, which is designed to take in a Pandas DataFrame, and based on data types, automatically generate all transformations and sub-models needed to generate expected predictions, arbitrary quantile predictions, and estimates of epistemic uncertainty.

### FAQ

**Why is the package named "tabularasa"?**

- I'm not a strong proponent of the tabula rasa theory of development, I just wanted a name with "tabular" in it.  Plus, I like that tabula rasa hints at the ability to learn anything, which ideally (although not practically) our models could.

**What is the long-term plan for tabularasa?**

- Ideally (in order of how much I care about them):
  - Continue to improve and evolve the default network that's used when a specific network isn't specified.
  - Better software engineering practices (tests, error messages, etc.).
  - Potentially expand into problems beyond regression.  It isn't necessarily clear how to generalize monotonicity for multiclass classification problems, but I'd be interested if others see value here.

### TODO

- Clean up wasted memory usage in `TabulaRasaRegressor()`
- Make save and reload effortless
- Generate partial dependence plots within the library
- Get GPU working
- Write basic unit tests
- Allow for networks with all the combinations of monotonic, non-monotonic, and categorical features
- Publish in PyPI
