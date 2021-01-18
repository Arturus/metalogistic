import numpy as np
from scipy import optimize
from scipy import stats
import matplotlib.pyplot as plt
import warnings
from . import support

cache = {}

class MetaLogistic(stats.rv_continuous):
	'''
	The only class in this package.

	We subclass scipy.stats.rv_continuous so we can make use of all the nice SciPy methods. We redefine the private methods
	_cdf, _pdf, and _ppf, and SciPy will make calls to these whenever needed.
	'''
	def __init__(self,
				 cdf_ps=None,
				 cdf_xs=None,
				 term=None,
				 fit_method=None,
				 lbound=None,
				 ubound=None,
				 a_vector=None,
				 feasibility_method='SmallMReciprocal',
				 validate_inputs=True):
		'''
		You must either provide CDF data or directly provide an a-vector. All other parameters are optional.

		:param cdf_ps: Probabilities of the CDF input data.
		:param cdf_xs: X-values of the CDF input data (the pre-images of the probabilities).
		:param term: Produces a `term`-term metalog. Cannot be greater than the number of CDF points provided. By default, it is equal to the number of CDF points.
		:param fit_method: Set to 'Linear least squares' to allow linear least squares only. By default, numerical methods are tried if linear least squares fails.
		:param lbound: Lower bound
		:param ubound: Upper bound
		:param a_vector: You may supply the a-vector directly, in which case the input data `cdf_ps` and `cdf_xs` are not used for fitting.
		:param feasibility_method: The method used to determine whether an a-vector corresponds to a feasible (valid) probability distribution. Its most important use is in the numerical solver, where it can have an impact on peformance and correctness. The options are: 'SmallMReciprocal' (default),'QuantileSumNegativeIncrements','QuantileMinimumIncrement'.
		'''
		super(MetaLogistic, self).__init__()

		if validate_inputs:
			self.validateInputs(
				cdf_ps=cdf_ps,
				cdf_xs=cdf_xs,
				term=term,
				fit_method=fit_method,
				lbound=lbound,
				ubound=ubound,
				a_vector=a_vector,
				feasibility_method=feasibility_method
			)

		if lbound == -np.inf:
			print("Infinite lower bound was ignored")
			lbound = None

		if ubound == np.inf:
			print("Infinite upper bound was ignored")
			ubound = None

		if lbound is None and ubound is None:
			self.boundedness = False
		if lbound is None and ubound is not None:
			self.boundedness = 'upper'
			self.a , self.b = -np.inf, ubound
		if lbound is not None and ubound is None:
			self.boundedness = 'lower'
			self.a, self.b = lbound, np.inf
		if lbound is not None and ubound is not None:
			self.boundedness = 'bounded'
			self.a, self.b = lbound, ubound
		self.lbound = lbound
		self.ubound = ubound

		self.fit_method_requested = fit_method
		self.numeric_ls_solver_used = None
		self.feasibility_method = feasibility_method

		self.cdf_ps = cdf_ps
		self.cdf_xs = cdf_xs
		if cdf_xs is not None and cdf_ps is not None:
			self.cdf_len = len(cdf_ps)
			self.cdf_ps = np.asarray(self.cdf_ps)
			self.cdf_xs = np.asarray(self.cdf_xs)

		# Special case where a MetaLogistic object is created by supplying the a-vector directly.
		if a_vector is not None:
			self.a_vector = a_vector
			if term is None:
				self.term = len(a_vector)
			else:
				self.term = term
			return


		if term is None:
			self.term = self.cdf_len
		else:
			self.term = term


		self.constructZVec()
		self.constructYMatrix()

		#  Try linear least squares
		self.fitLinearLeastSquares()
		self.fit_method_used = 'Linear least squares'

		# If linear least squares result is feasible
		if self.isFeasible():
			self.valid_distribution = True

		#  If linear least squares result is not feasible
		else:
			# if the user allows it, use numerical least squares
			if not fit_method == 'Linear least squares':
				self.fit_method_used = 'numeric'
				# TODO: set a timeout (e.g. 1 second) for the call to fitNumericLeastSquares(). If the call
				# times out with the default feasibility method, iterate through all possible methods,
				# keeping the best result. This is because, in my experience, if a method will succeed, it succeeds
				# within hundreds of milliseconds; if it's been going on for more than a second, another method will likely give
				# good results faster.
				with warnings.catch_warnings():
					warnings.filterwarnings("ignore", category=UserWarning, module='scipy.optimize')
					self.fitNumericLeastSquares(feasibility_method=self.feasibility_method)

			# If only LLS is allowed, we cannot find a valid metalog
			else:
				self.valid_distribution = False

		if not self.isFeasible():
			print("Warning: the program was not able to fit a valid metalog distribution for your data.")


	def validateInputs(self,
			cdf_ps,
			cdf_xs,
			term,
			fit_method,
			lbound,
			ubound,
			a_vector,
			feasibility_method):

		def checkPsXs(array, name):
			if support.isListLike(array):
				for item in array:
					if not support.isNumeric(item):
						raise ValueError(name+" must be an array of numbers")
			else:
				raise ValueError(name + " must be an array of numbers")

		if cdf_xs is not None:
			checkPsXs(cdf_xs, 'cdf_xs')
			cdf_xs = np.asarray(cdf_xs)

		if cdf_ps is not None:
			checkPsXs(cdf_ps,'cdf_ps')
			cdf_ps = np.asarray(cdf_ps)

			if np.any(cdf_ps<0) or np.any(cdf_ps>1):
				raise ValueError("Probabilities must be between 0 and 1")

		if cdf_ps is not None and cdf_xs is not None:
			if len(cdf_ps) != len(cdf_xs):
				raise ValueError("cdf_ps and cdf_xs must have the same length")

			if len(cdf_ps)<2:
				raise ValueError("Must provide at least two CDF data points")

			ps_xs_sorted = sorted(zip(cdf_ps, cdf_xs))
			prev = -np.inf
			for tuple in ps_xs_sorted:
				p,x = tuple
				if x<=prev:
					print("Warning: Non-increasing CDF input data. Are you sure?")
				prev = x

			if term is not None:
				if term > len(cdf_ps):
					raise ValueError("term cannot be greater than the number of CDF data points provided")

		if term is not None:
			if term<3:
				raise ValueError("term cannot be less than 3. Just use the logistic distribution, no need to go meta!")

		if a_vector is not None and term is not None:
			if term>len(a_vector):
				raise ValueError("term cannot be greater than the length of the a_vector")

		if fit_method is not None:
			if fit_method not in ['Linear least squares']:
				raise ValueError("Unknown fit method")

		if lbound is not None:
			if lbound>min(cdf_xs):
				raise ValueError("Lower bound cannot be greater than the lowest data point")

		if ubound is not None:
			if ubound < max(cdf_xs):
				raise ValueError("Upper bound cannot be less than the greatest data point")

		feasibility_methods = ['SmallMReciprocal','QuantileSumNegativeIncrements','QuantileMinimumIncrement']
		if not feasibility_method in feasibility_methods:
			raise ValueError("feasibility_method must be one of: "+str(feasibility_methods))

	def isFeasible(self):
		if self.feasibility_method == 'QuantileMinimumIncrement':
			s = self.QuantileMinimumIncrement()
			if s<0:
				self.valid_distribution_violation = s
				self.valid_distribution = False
			else:
				self.valid_distribution_violation = 0
				self.valid_distribution = True

		if self.feasibility_method == 'QuantileSumNegativeIncrements':
			s = self.infeasibilityScoreQuantileSumNegativeIncrements()
			self.valid_distribution_violation = s
			self.valid_distribution = s==0

		if self.feasibility_method == 'SmallMReciprocal':
			s = self.infeasibilityScoreSmallMReciprocal()
			self.valid_distribution_violation = s
			self.valid_distribution = s==0

		return self.valid_distribution


	def fitLinearLeastSquares(self):
		'''
		Constructs the a-vector by linear least squares, as defined in Keelin 2016, Equation 7 (unbounded case), Equation 12 (semi-bounded and bounded cases).
		'''
		left = np.linalg.inv(np.dot(self.YMatrix.T, self.YMatrix))
		right = np.dot(self.YMatrix.T, self.z_vec)

		self.a_vector = np.dot(left, right)

	def fitNumericLeastSquares(self, feasibility_method, avoid_extreme_steepness=True):
		'''
		Constructs the a-vector by attempting to approximate, using numerical methods, the feasible a-vector that minimizes least squares on the CDF.

		`feasibility_method` is the method by which we check whether an a-vector is feasible. It often has an important impact on the correctness and speed
		of the numerical minimization results.
		'''
		bounds_kwargs = {}
		if self.lbound is not None:
			bounds_kwargs['lbound'] = self.lbound
		if self.ubound is not None:
			bounds_kwargs['ubound'] = self.ubound

		def loss_function(a_candidate):
			# Setting a_vector in this MetaLogistic call overrides the cdf_ps and cdf_xs arguments, which are only used
			# for meanSquareError().
			return MetaLogistic(self.cdf_ps, self.cdf_xs, **bounds_kwargs, a_vector=a_candidate, validate_inputs=False).meanSquareError()

		# Choose the method of determining feasibility.
		def feasibilityViaCDFSumNegative(a_candidate):
			return MetaLogistic(a_vector=a_candidate, **bounds_kwargs, validate_inputs=False).infeasibilityScoreQuantileSumNegativeIncrements()

		def feasibilityViaQuantileMinimumIncrement(a_candidate):
			return MetaLogistic(a_vector=a_candidate, **bounds_kwargs, validate_inputs=False).QuantileMinimumIncrement()

		def feasibilityViaSmallMReciprocal(a_candidate):
			return MetaLogistic(a_vector=a_candidate, **bounds_kwargs, validate_inputs=False).infeasibilityScoreSmallMReciprocal()

		if feasibility_method == 'SmallMReciprocal':
			def feasibilityBool(a_candidate):
				return feasibilityViaSmallMReciprocal(a_candidate) == 0
			feasibility_constraint = optimize.NonlinearConstraint(feasibilityViaSmallMReciprocal, 0, 0)

		if feasibility_method == 'QuantileSumNegativeIncrements':
			def feasibilityBool(a_candidate):
				return feasibilityViaCDFSumNegative(a_candidate) == 0
			feasibility_constraint = optimize.NonlinearConstraint(feasibilityViaCDFSumNegative, 0, 0)

		if feasibility_method == 'QuantileMinimumIncrement':
			def feasibilityBool(a_candidate):
				return feasibilityViaQuantileMinimumIncrement(a_candidate) >= 0
			feasibility_constraint = optimize.NonlinearConstraint(feasibilityViaQuantileMinimumIncrement, 0, np.inf)

		shifted = self.findShiftedValue((tuple(self.cdf_ps),tuple(self.cdf_xs),self.lbound,self.ubound), cache)
		if shifted:
			optimize_result,shift_distance = shifted
			self.a_vector = np.append(optimize_result.x[0]+shift_distance, optimize_result.x[1:])
			self.numeric_leastSQ_OptimizeResult = None
			if avoid_extreme_steepness: self.avoidExtremeSteepness()
			return

		a0 = self.a_vector

		# First, try the default solver, which is often fast and accurate
		options = {}
		optimize_results = optimize.minimize(loss_function,
											 a0,
											 constraints=feasibility_constraint,
											 options=options)
		self.numeric_ls_solver_used = 'Default'

		# If the mean square error is too large or distribution invalid, try the trust-constr solver
		if optimize_results.fun > 0.01 or not feasibilityBool(optimize_results.x):
			options = {'xtol': 1e-6,  # this improves speed considerably vs the default of 1e-8
					   'maxiter': 300  # give up if taking too long
					   }
			optimize_results_alternate = optimize.minimize(loss_function,
												 a0,
												 constraints=feasibility_constraint,
												 method='trust-constr',
												 options=options)
			self.numeric_ls_solver_used = 'trust-constr'

			if optimize_results_alternate.constr_violation == 0:
				if optimize_results_alternate.fun < optimize_results.fun:
					optimize_results = optimize_results_alternate
				else:
					optimize_results = optimize_results

		cache[(tuple(self.cdf_ps), tuple(self.cdf_xs), self.lbound, self.ubound)] = optimize_results
		self.a_vector = optimize_results.x
		if avoid_extreme_steepness: self.avoidExtremeSteepness()
		self.numeric_leastSQ_OptimizeResult = optimize_results

	def avoidExtremeSteepness(self):
		'''
		Since we are using numerical approximations to determine feasibility,
		the feasible distribution that most closely fits the data may actually be just barely infeasible.

		A hint that this is the case is that an extremely large spike in PDF will result,
		with densities in the tens of thousands! (If we were able to evaluate the PDF
		everywhere, we would see that it is actually negative in a tiny region).

		We can avoid secretly-infeasible a-vectors by biasing the *last* a-parameter very slightly (by 1%) towards zero when
		PDF steepness is extreme, which will usually guarantee that the distribution is feasible,
		at a very small cost to goodness of fit.
		'''
		steepness = self.pdfMax()
		if steepness > 10:
			self.a_vector = np.append(self.a_vector[0:-1], self.a_vector[-1] * 99 / 100)

	def pdfMax(self):
		'''
		Find a very rough approximation of the max of the PDF. Used for penalizing extremely steep distributions.
		'''
		check_ps_from = 0.001
		number_to_check = 100
		ps_to_check = np.linspace(check_ps_from, 1 - check_ps_from, number_to_check)
		return max(self.densitySmallM(ps_to_check))

	@staticmethod
	def findShiftedValue(input_tuple, cache):
		if not cache:
			return False
		for cache_tuple, cache_value in cache.items():
			shifted = MetaLogistic.isSameShifted(support.tupleToDict(cache_tuple), support.tupleToDict(input_tuple))
			if shifted is not False:
				return cache_value,shifted
		return False

	@staticmethod
	def isSameShifted(dict1,dict2):
		bounds = [dict1['lbound'], dict2['lbound'], dict1['ubound'], dict2['ubound']]
		if any([i is not None for i in bounds]):
			return False

		if not dict1['cdf_ps'] == dict2['cdf_ps']:
			return False

		dict1Xsorted = sorted(dict1['cdf_xs'])
		dict2Xsorted = sorted(dict2['cdf_xs'])

		diffdelta = np.abs(np.diff(dict1Xsorted) - np.diff(dict2Xsorted))
		diffdelta_relative = diffdelta/dict1Xsorted[1:]
		if np.all(diffdelta_relative < .005): # I believe this is necessary because of the imprecision of dragging in d3.js
			return dict2Xsorted[0]-dict1Xsorted[0]
		else:
			return False

	def meanSquareError(self):
		ps_on_fitted_cdf = self.cdf(self.cdf_xs)
		sum_sq_error = np.sum((self.cdf_ps - ps_on_fitted_cdf) ** 2)
		return sum_sq_error/self.cdf_len

	def infeasibilityScoreQuantileSumNegativeIncrements(self):
		check_ps_from = 0.001
		number_to_check = 200  # This parameter is very important to both performance and correctness.
		ps_to_check = np.linspace(check_ps_from, 1 - check_ps_from, number_to_check)
		xs_to_check = self.quantile(ps_to_check)
		prev = -np.inf
		infeasibility_score = 0
		for item in xs_to_check:
			diff = item - prev
			if diff < 0:
				# Logarithm of the difference, to keep this scale-free
				infeasibility_score += np.log(1-diff)
			prev = item
		return infeasibility_score

	def infeasibilityScoreSmallMReciprocal(self):
		'''
		Checks whether the a-vector is feasible using the reciprocal of the small-m PDF, as defined in Keelin 2016, Equation 5 (see also equation 10).
		This check is performed for an array of probabilities; all reciprocals-of-densities that are negative (corresponding to an invalid distribution, since a density
		must be non-negative) are summed together into a score that attempts to quantify to what degree the a-vector is infeasible. This captures how much of the
		PDF is negative, and how far below 0 it is.

		As Keelin notes, Equations 5 and 10 are equivalent, i.e. reciprocals of densities will be positive whenever densities are positive, and so this method would return
		0 for all and only the same inputs, if we checked densities instead of reciprocals (i.e. if we omitted `densities_reciprocal = 1/densities_to_check`). However, using the reciprocal
		seems to be more amenable to finding feasible a-vectors with scipy.optimize.minimize; this might be because the reciprocal (Equation 5) is linear in the elements of the
		a-vector.
		'''
		check_ps_from = 0.001
		number_to_check = 100  # Some light empirical testing suggests 100 may be a good value here.
		ps_to_check = np.linspace(check_ps_from, 1 - check_ps_from, number_to_check)

		densities_to_check = self.densitySmallM(ps_to_check)
		densities_reciprocal = 1/densities_to_check
		infeasibility_score = np.abs(np.sum(densities_reciprocal[densities_reciprocal<0]))

		return infeasibility_score

	def QuantileSlopeNumeric(self, p):
		'''
		Gets the slope of the quantile function by simple finite-difference approximation.
		'''
		epsilon = 1e-5
		if not np.isfinite(self.quantile(p+epsilon)):
			epsilon = -epsilon
		cdfSlope = optimize.approx_fprime(p,self.quantile,epsilon)
		return cdfSlope

	def QuantileMinimumIncrement(self):
		'''
		Nice idea but in practise the worst feasibility method, I might remove it.
		'''
		# Get a good initial guess
		check_ps_from = 0.001
		number_to_check = 100
		ps_to_check = np.linspace(check_ps_from, 1 - check_ps_from, number_to_check)
		xs = self.quantile(ps_to_check)
		xs_diff = np.diff(xs)
		i = np.argmin(xs_diff)
		p0 = ps_to_check[i]

		# Do the minimization
		r = optimize.minimize(self.QuantileSlopeNumeric, x0=p0, bounds=[(0, 1)])
		return r.fun

	def constructZVec(self):
		'''
		Constructs the z-vector, as defined in Keelin 2016, Section 3.3 (unbounded case, where it is called the `x`-vector),
		Section 4.1 (semi-bounded case), and Section 4.3 (bounded case).

		This vector is a transformation of cdf_xs to account for bounded or semi-bounded distributions.
		When the distribution is unbounded, the z-vector is simply equal to cdf_xs.
		'''
		if not self.boundedness:
			self.z_vec = self.cdf_xs
		if self.boundedness == 'lower':
			self.z_vec = np.log(self.cdf_xs-self.lbound)
		if self.boundedness == 'upper':
			self.z_vec = -np.log(self.ubound - self.cdf_xs)
		if self.boundedness == 'bounded':
			self.z_vec = np.log((self.cdf_xs-self.lbound)/(self.ubound-self.cdf_xs))

	def constructYMatrix(self):
		'''
		Constructs the Y-matrix, as defined in Keelin 2016, Equation 8.
		'''

		# The series of Y_n matrices. Although we only use the last matrix in the series, the entire series is necessary to construct it
		Y_ns = {}
		ones = np.ones(self.cdf_len).reshape(self.cdf_len, 1)
		column_2 = np.log(self.cdf_ps / (1 - self.cdf_ps)).reshape(self.cdf_len, 1)
		column_4 = (self.cdf_ps - 0.5).reshape(self.cdf_len, 1)
		Y_ns[2] = np.hstack([ones, column_2])
		Y_ns[3] = np.hstack([Y_ns[2], column_4 * column_2])
		Y_ns[4] = np.hstack([Y_ns[3], column_4])

		if (self.term > 4):
			for n in range(5, self.term + 1):
				if n % 2 != 0:
					new_column = column_4 ** ((n - 1) / 2)
					Y_ns[n] = np.hstack([Y_ns[n - 1], new_column])

				if n % 2 == 0:
					new_column = (column_4 ** (n / 2 - 1)) * column_2
					Y_ns[n] = np.hstack([Y_ns[n - 1], new_column])

		self.YMatrix = Y_ns[self.term]

	def quantile(self, probability, force_unbounded=False):
		'''
		The metalog inverse CDF, or quantile function, as defined in Keelin 2016, Equation 6 (unbounded case), Equation 11 (semi-bounded case),
		and Equation 14 (bounded case).

		`probability` must be a scalar.
		'''

		# if not 0 <= probability <= 1:
		# 	raise ValueError("Probability in call to quantile() must be between 0 and 1")

		if support.isListLike(probability):
			return np.asarray([self.quantile(i) for i in probability])

		if probability <= 0:
			if (self.boundedness == 'lower' or self.boundedness == 'bounded') and not force_unbounded:
				return self.lbound
			else:
				return -np.inf

		if probability >= 1:
			if (self.boundedness == 'upper' or self.boundedness == 'bounded') and not force_unbounded:
				return self.ubound
			else:
				return np.inf

		# `self.a_vector` is 0-indexed, while in Keelin 2016 the a-vector is 1-indexed.
		# To make this method as easy as possible to read if following along with the paper, I create a dictionary `a`
		# that mimics a 1-indexed vector.
		a = {i + 1: element for i, element in enumerate(self.a_vector)}

		# The series of quantile functions. Although we only return the last result in the series, the entire series is necessary to construct it
		ln_p_term = np.log(probability / (1 - probability))
		p05_term = probability - 0.5
		quantile_functions = {}

		quantile_functions[2] = a[1] + a[2] * ln_p_term
		if self.term>2:
			quantile_functions[3] = quantile_functions[2] + a[3] * p05_term * ln_p_term
		if self.term>3:
			quantile_functions[4] = quantile_functions[3] + a[4] * p05_term

		if (self.term > 4):
			for n in range(5, self.term + 1):
				if n % 2 != 0:
					quantile_functions[n] = quantile_functions[n - 1] + a[n] * p05_term ** ((n - 1) / 2)

				if n % 2 == 0:
					quantile_functions[n] = quantile_functions[n - 1] + a[n] * p05_term ** (n / 2 - 1) * ln_p_term

		quantile_function = quantile_functions[self.term]

		if not force_unbounded:
			if self.boundedness == 'lower':
				quantile_function = self.lbound + np.exp(quantile_function)  # Equation 11
			if self.boundedness == 'upper':
				quantile_function = self.ubound - np.exp(-quantile_function)
			if self.boundedness == 'bounded':
				quantile_function = (self.lbound+self.ubound*np.exp(quantile_function))/(1+np.exp(quantile_function))  # Equation 14

		return quantile_function

	def densitySmallM(self,cumulative_prob,force_unbounded=False):
		'''
		This is the metalog PDF as a function of cumulative probability, as defined in Keelin 2016, Equation 9 (unbounded case),
		Equation 13 (semi-bounded case), Equation 15 (bounded case).

		Notice the unusual definition of the PDF, which is why I call this function densitySmallM in reference to the notation in
		Keelin 2016.
		'''

		if support.isListLike(cumulative_prob):
			return np.asarray([self.densitySmallM(i) for i in cumulative_prob])

		if not 0 <= cumulative_prob <= 1:
			raise ValueError("Probability in call to densitySmallM() must be between 0 and 1")
		if not self.boundedness and (cumulative_prob==0 or cumulative_prob==1):
			raise ValueError("Probability in call to densitySmallM() cannot be equal to 0 and 1 for an unbounded distribution")


		# The series of density functions. Although we only return the last result in the series, the entire series is necessary to construct it
		density_functions = {}

		# `self.a_vector` is 0-indexed, while in Keelin 2016 the a-vector is 1-indexed.
		# To make this method as easy as possible to read if following along with the paper, I create a dictionary `a`
		# that mimics a 1-indexed vector.
		a = {i + 1: element for i, element in enumerate(self.a_vector)}

		ln_p_term = np.log(cumulative_prob / (1 - cumulative_prob))
		p05_term = cumulative_prob - 0.5
		p1p_term = cumulative_prob*(1-cumulative_prob)

		density_functions[2] = p1p_term/a[2]
		if self.term>2:
			density_functions[3] = 1/(1/density_functions[2] + a[3]*(p05_term/p1p_term+ln_p_term))
		if self.term>3:
			density_functions[4] = 1/(1/density_functions[3] + a[4])

		if (self.term > 4):
			for n in range(5, self.term + 1):
				if n % 2 != 0:
					density_functions[n] = 1/(1/density_functions[n-1]+ a[n]*((n-1)/2)*p05_term**((n-3)/2))

				if n % 2 == 0:
					density_functions[n] = 1/(1/density_functions[n-1] + a[n]*(p05_term**(n/2-1)/p1p_term +
																			  (n/2-1)*p05_term**(n/2-2)*ln_p_term)
											  )

		density_function = density_functions[self.term]
		if not force_unbounded:
			if self.boundedness == 'lower':   # Equation 13
				if 0<cumulative_prob<1:
					density_function = density_function * np.exp(-self.quantile(cumulative_prob, force_unbounded=True))
				elif cumulative_prob == 0:
					density_function = 0
				else:
					raise ValueError("Probability in call to densitySmallM() cannot be equal to 1 with a lower-bounded distribution.")

			if self.boundedness == 'upper':
				if 0 < cumulative_prob < 1:
					density_function = density_function * np.exp(self.quantile(cumulative_prob, force_unbounded=True))
				elif cumulative_prob == 1:
					density_function = 0
				else:
					raise ValueError("Probability in call to densitySmallM() cannot be equal to 0 with a upper-bounded distribution.")

			if self.boundedness == 'bounded':  # Equation 15
				if 0 < cumulative_prob < 1:
					x_unbounded = np.exp(self.quantile(cumulative_prob, force_unbounded=True))
					density_function = density_function * (1 + x_unbounded)**2 / ((self.ubound - self.lbound) * x_unbounded)
				if cumulative_prob==0 or cumulative_prob==1:
					density_function = 0

		return density_function

	def getCumulativeProb(self, x):
		'''
		The metalog is defined in terms of its inverse CDF or quantile function. In order to get probabilities for a given x-value,
		like in a traditional CDF, we invert this quantile function using a numerical equation solver.

		`x` must be a scalar
		'''
		f_to_zero = lambda probability: self.quantile(probability) - x
		return optimize.brentq(f_to_zero, 0, 1, disp=True)

	def _cdf(self, x):
		'''
		This is where we override the SciPy method for the CDF.

		`x` may be a scalar or list-like.
		'''
		if support.isListLike(x):
			return [self._cdf(i) for i in x]
		if support.isNumeric(x):
			return self.getCumulativeProb(x)

	def _ppf(self, probability):
		'''
		This is where we override the SciPy method for the inverse CDF or quantile function (ppf stands for percent point function).

		`probability` may be a scalar or list-like.
		'''
		return self.quantile(probability)


	def _pdf(self, x):
		'''
		This is where we override the SciPy method for the PDF.

		`x` may be a scalar or list-like.
		'''
		if support.isListLike(x):
			return [self._pdf(i) for i in x]

		if support.isNumeric(x):
			cumulative_prob = self.getCumulativeProb(x)
			return self.densitySmallM(cumulative_prob)

	def printSummary(self):
		print("Fit method used:", self.fit_method_used)
		print("Distribution is valid:", self.valid_distribution)
		print("Method for determining distribution validity:", self.feasibility_method)
		if not self.valid_distribution:
			print("Distribution validity constraint violation:", self.valid_distribution_violation)
		if not self.fit_method_used == 'Linear least squares':
			print("Solver for numeric fit:", self.numeric_ls_solver_used)
			print("Solver convergence:", self.numeric_leastSQ_OptimizeResult.success)
			# print("Solver convergence message:", self.numeric_leastSQ_OptimizeResult.message)
		print("Mean square error:", self.meanSquareError())
		print('a vector:', self.a_vector)

	def createCDFPlotData(self,p_from_to=(.001,.999), x_from_to=(None,None),n=100):
		p_from, p_to = p_from_to
		x_from, x_to = x_from_to
		if x_from is not None and x_to is not None:
			p_from = self.getCumulativeProb(x_from)
			p_to = self.getCumulativeProb(x_to)

		cdf_ps = np.linspace(p_from,p_to,n)
		cdf_xs = self.quantile(cdf_ps)

		return {'X-values':cdf_xs,'Probabilities':cdf_ps}

	def createPDFPlotData(self, p_from_to=(.001,.999), x_from_to=(None,None), n=100):
		p_from, p_to = p_from_to
		x_from, x_to = x_from_to
		if x_from is not None and x_to is not None:
			p_from = self.getCumulativeProb(x_from)
			p_to = self.getCumulativeProb(x_to)

		pdf_ps = np.linspace(p_from, p_to, n)
		pdf_xs = self.quantile(pdf_ps)
		pdf_densities = self.densitySmallM(pdf_ps)

		return {'X-values': pdf_xs, 'Densities': pdf_densities}

	def displayPlot(self, p_from_to=(.001,.999), x_from_to=(None,None), n=100, hide_extreme_densities=50):
		'''
		The parameter `hide_extreme_densities` is used on the PDF plot, to set its y-axis maximum to `hide_extreme_densities` times
		the median density. This is because, when given extreme input data, the resulting metalog might have very short but extremely tall spikes in the PDF
		(where the maximum density might be in the hundreds), which would make the PDF plot unreadable if the entire spike was included.

		If both `p_from_to` and `x_from_to` are specified, `p_from_to` is overridden.
		'''

		p_from, p_to = p_from_to
		x_from, x_to = x_from_to
		if x_from is not None and x_to is not None:
			p_from = self.getCumulativeProb(x_from)
			p_to = self.getCumulativeProb(x_to)

		fig, (cdf_axis, pdf_axis) = plt.subplots(2)
		# fig.set_size_inches(6, 6)

		cdf_data = self.createCDFPlotData(p_from_to=(p_from,p_to),n=n)
		cdf_axis.plot(cdf_data['X-values'],cdf_data['Probabilities'])
		if self.cdf_xs is not None and self.cdf_ps is not None:
			cdf_axis.scatter(self.cdf_xs, self.cdf_ps, marker='+', color='red')
		cdf_axis.set_title('CDF')

		pdf_data = self.createPDFPlotData(p_from_to=(p_from,p_to),n=n)
		pdf_axis.set_title('PDF')

		if hide_extreme_densities:
			density50 = np.percentile(pdf_data['Densities'],50)
			pdf_max_display = min(density50*hide_extreme_densities,1.05*max(pdf_data['Densities']))
			pdf_axis.set_ylim(top=pdf_max_display)

		pdf_axis.plot(pdf_data['X-values'], pdf_data['Densities'])

		fig.show()
		return fig

