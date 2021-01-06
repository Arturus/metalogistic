# The metalog distribution
This package is a Python implementation of the metalogistic or metalog distribution,
as described in [Keelin 2016](http://www.metalogdistributions.com/images/The_Metalog_Distributions_-_Keelin_2016.pdf).

The metalog is a highly flexible continuous univariate probability distribution that can be used to model data without traditional parameters.
Instead, the distribution is parametrized by points on a cumulative distribution function, and the metalog CDF usually passes through these points exactly.

The distribution is well suited to eliciting subjective probability distributions, and 
"can be used for real-time feedback to experts about the implications of their probability assessments". Its shape flexibility 
is "far beyond that of traditional distributions" and "enables 'the data to speak for itself' in contrast
to imposing unexamined and possibly inappropriate shape constraints on that data" (Keelin 2016).

# This package
This package:
* Provides an object-oriented interface.
* Returns a SciPy [continuous distribution object](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html),
which lets you use its many convenient, performant methods.
* Uses numerical methods to approximate a least-squares fit if the closed-form method described in Keelin 2016 fails, thus
allowing you to fit virtually any CDF data. 
 

# Usage
```python
from metalogistic import MetaLogistic
my_metalog = MetaLogistic(cdf_xs=[-5, 2, 20],cdf_ps=[.35,.5,.95])

my_metalog.cdf(10)
my_metalog.pdf(10)
my_metalog.ppf(0.8)
```

See also `example_usage.py`

# Installation 
```
pip install metalogistic
```