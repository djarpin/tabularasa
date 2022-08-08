from setuptools import setup

setup(name='tabularasa',
      version='0.1.1',
      description='Tabular PyTorch Neural Network with support for monotoric features and aleatoric and epistemic uncertainty estimates with a scikit-learn API',
      url='http://github.com/djarpin/tabularasa',
      author='David Arpin',
      author_email='arpin.david@gmail.com',
      license='BSD-3',
      packages=['tabularasa', 'tabularasa.gumnn'],
      zip_safe=False)
