# The metalog distribution
This package is a Python implementation of the **metalogistic or metalog** distribution,
as described in [Keelin 2016][k2016].

The metalog is a continuous univariate probability distribution that can be used to model data without traditional parameters.
Instead, the distribution is parametrized by **points on a cumulative distribution function (CDF)**, and the CDF of the
metalog fitted to these input points usually passes through them exactly.
The distribution can take almost any shape.

The distribution is well suited to **eliciting full subjective probability distributions** from a few
 CDF points. If used in this way, the result is a distribution that fits these points closely, without
 imposing strong shape constraints (as would be the case if fitting to a traditional distribution like the 
 normal or lognormal). [Keelin 2016][k2016] remarks that the metalog "can be used for real-time feedback to experts
 about the implications of their probability assessments".

See also the website [metalogdistributions.com](http://www.metalogdistributions.com/).

[k2016]: http://www.metalogdistributions.com/images/The_Metalog_Distributions_-_Keelin_2016.pdf

# This package
This package:
* Provides an object-oriented interface to instances of the class `MetaLogistic`
* Defines `MetaLogistic` as a subclass of SciPy [continuous distribution objects](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html),
which lets us use their many convenient, performant methods.
* Uses numerical methods to approximate a least-squares fit if the closed-form method described in Keelin 2016 fails.
 This allows us to fit an even wider range of CDF data. 
* Is fast (see `timings.py`).
 

# Usage

```python
from metalogistic import MetaLogistic

my_metalog = MetaLogistic(cdf_xs=[-5, 2, 20], cdf_ps=[.35, .5, .95])

# These methods can take scalars or arrays/lists/vectors
my_metalog.cdf(10)
my_metalog.pdf([10, 20, 21])
my_metalog.quantile([0.8, .99])

# These methods conveniently display useful information
my_metalog.print_summary()
my_metalog.display_plot()
```

See also `example_usage.py`

# Installation 
```
pip install metalogistic
```

# Speed
`timings.py`

When using linear least squares:
```
#### Speed test ####
Data:
cdf_ps [0.15, 0.5, 0.9]
cdf_xs [-20, -1, 40]
Bounds: None None

Timings:
'doFit'  17.08 ms
'createPlotData'  6.98 ms
```

When we are forced to use numerical fitting methods:
```
#### Speed test ####
Data:
cdf_ps [0.15, 0.5, 0.9]
cdf_xs [-20, -1, 100]
Bounds: None 1000

Timings:
'doFit'  600.44 ms
'createPlotData'  9.97 ms

#### Speed test ####
Data:
cdf_ps [0.15, 0.5, 0.9]
cdf_xs [-20, -1, 100]
Bounds: None None

Timings:
'doFit'  1035.46 ms
'createPlotData'  6.98 ms
```

# License
If AGPL is a problem for you, please [contact me](https://tadamcz.com/). As I am currently the sole author, we can probably work something out.
