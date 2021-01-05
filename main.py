import numpy as np
from scipy import optimize
from scipy import stats

class MetaLogistic(stats.rv_continuous):
	'''
	We subclass scipy.stats.rv_continuous so we can make use of all the nice SciPy methods. We redefine the private methods
	_cdf, _pdf, and _ppf, and SciPy will make calls to these whenever needed.
	'''
	def __init__(self, cdf_ps, cdf_xs, term=None, fit_method=None, lbound=None, ubound=None):
		'''
		:param cdf_ps: Probabilities of the CDF input data.
		:param cdf_xs: X-values of the CDF input data (the pre-images of the probabilities).
		:param term: Produces a `term`-term metalog. Cannot be greater than the number of CDF points provided. By default, it is
		equal to the number of CDF points.
		:param fit_method:
		:param boundedness:
		:param bounds:
		'''
		super(MetaLogistic, self).__init__()

		if lbound is None and ubound is None:
			self.boundedness = False
		if lbound is None and ubound is not None:
			self.boundedness = 'upper'
			self.ubound = ubound
		if lbound  is not None and ubound is None:
			self.boundedness = 'lower'
			self.lbound = lbound
		if lbound is not None and ubound is not None:
			self.boundedness = 'bounded'
			self.lbound = lbound
			self.ubound = ubound

		self.fit_method_requested = fit_method
		self.cdf_ps = np.asarray(cdf_ps)
		self.cdf_xs = np.asarray(cdf_xs)
		self.cdf_len = len(cdf_ps)

		if len(cdf_ps) != len(cdf_ps):
			raise ValueError("cdf_ps and cdf_xs must have the same length")

		if term is None:
			term = self.cdf_len
		self.term = term

		self.constructZVec()
		self.constructYMatrix()
		self.fitLinearLeastSquares()

	def fitLinearLeastSquares(self):
		'''
		Constructs the a-vector by linear least squares, as defined in Keelin 2016, Equation 7 (unbounded case), Equation 12 (semi-bounded case).

		'''
		left = np.linalg.inv(np.dot(self.YMatrix.T, self.YMatrix))
		right = np.dot(self.YMatrix.T, self.z_vec)

		self.a_vector = np.dot(left, right)

	def constructZVec(self):
		'''
		Constructs the z-vector, as defined in Keelin 2016, Section 3.3. (unbounded case, where it is called the `x`-vector),
		Section 4.1 (semi-bounded case), and Section 4.3 (bounded case).

		This vector is a transformation of cdf_xs to account for bounded or semi-bounded distributions.
		When the distribution is unbounded, the z-vector is simply equal to cdf_xs.
		'''
		if not self.boundedness:
			self.z_vec = self.cdf_xs
		if self.boundedness == 'lower':
			self.z_vec = np.log(self.cdf_xs-self.lbound)
		if self.boundedness == 'upper':
			self.z_vec = np.log(self.ubound - self.cdf_xs)
		if self.boundedness == 'bounded':
			self.z_vec = np.log((self.cdf_xs-self.lbound)/(self.ubound-self.cdf_xs))

	def constructYMatrix(self):
		'''
		Constructs the Y-matrix, as defined in Keelin 2016, Equation 8.
		'''

		# The series of Y_n matrices. Although we only return the last matrix in the series, the entire series is necessary to construct it
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

	def quantile(self, probability):
		'''
		The metalog inverse CDF, or quantile function, as defined in Keelin 2016, Equation 6 (unbounded case), Equation 11 (semi-bounded case),
		and Equation 14 (bounded case).

		`probability` must be a scalar.
		'''

		if not 0 <= probability <= 1:
			raise ValueError("Probability in call to quantile() must be between 0 and 1")

		if probability == 0:
			if self.boundedness == 'lower' or self.boundedness == 'bounded':
				return self.lbound
			else:
				return -np.inf

		if probability == 1:
			if self.boundedness == 'upper' or self.boundedness == 'bounded':
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

		if self.boundedness == 'lower':
			quantile_function = self.lbound + np.exp(quantile_function)  # Equation 11
		if self.boundedness == 'upper':
			quantile_function = self.ubound - np.exp(-quantile_function)
		if self.boundedness == 'bounded':
			quantile_function = (self.lbound+self.ubound*np.exp(quantile_function))/(1+np.exp(quantile_function))  # Equation 14

		return quantile_function

	def densitySmallM(self,cumulative_prob):
		'''
		This is the metalog PDF as a function of cumulative probability, as defined in Keelin 2016, Equation 9 (unbounded case),
		Equation 13 (semi-bounded case),
		Notice the unusual definition of the PDF, which is why I call this function densitySmallM in reference to the notation in
		Keelin 2016.
		'''

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

		density_functions[2] = cumulative_prob*(1-cumulative_prob)/a[2]
		density_functions[3] = 1/(1/density_functions[2] + a[3]*(p05_term/p1p_term)+ln_p_term)
		if self.term>3:
			density_functions[4] = 1/(1/density_functions[3] + a[4])

		if (self.term > 4):
			for n in range(5, self.term + 1):
				if n % 2 != 0:
					density_functions[n] = 1/(1/density_functions[n-1]+ a[n]*((n-1)/2)*p05_term**((n-3)/2))

				if n % 2 == 0:
					density_functions[n] = 1/(1/density_functions[n-1] + a[n](p05_term**(n/2-1)/p1p_term +
																			  (n/2-1)*p05_term**(n/2-2)*ln_p_term)
											  )

		density_function = density_functions[self.term]

		if self.boundedness == 'lower':   # Equation 13
			if 0<cumulative_prob<1:
				density_function = density_function * np.exp(-self.quantile(cumulative_prob))
			elif cumulative_prob == 0:
				density_function = 0
			else:
				raise ValueError("Probability in call to densitySmallM() cannot be equal to 1 with a lower-bounded distribution.")
		if self.boundedness == 'upper':
			if 0 < cumulative_prob < 1:
				density_function = density_function * np.exp(self.quantile(cumulative_prob))
			elif cumulative_prob == 1:
				density_function = 0
			else:
				raise ValueError("Probability in call to densitySmallM() cannot be equal to 0 with a upper-bounded distribution.")

		if self.boundedness == 'bounded':  # Equation 15
			if 0 < cumulative_prob < 1:
				density_function = density_function * ((1+np.exp(self.quantile(cumulative_prob)))**2) / ((self.ubound-self.lbound)*np.exp(self.quantile(cumulative_prob)))
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
		if self.isListLike(x):
			return [self._cdf(i) for i in x]
		if self.isNumeric(x):
			return self.getCumulativeProb(x)

	def _ppf(self, probability):
		'''
		This is where we override the SciPy method for the inverse CDF or quantile function (ppf stands for percent point function)

		`probability` may be a scalar or list-like.
		'''
		if self.isListLike(probability):
			return [self._ppf(i) for i in probability]

		if self.isNumeric(probability):
			return self.quantile(probability)

	def _pdf(self, x):
		'''
		This is where we override the SciPy method for the PDF.

		`x` may be a scalar or list-like.
		'''
		if self.isListLike(x):
			return [self._pdf(i) for i in x]

		if self.isNumeric(x):
			cumulative_prob = self.getCumulativeProb(x)
			return self.densitySmallM(cumulative_prob)

	def isNumeric(self,object):
		return isinstance(object, (float, int)) or (isinstance(object,np.ndarray) and object.ndim==0)

	def isListLike(self,object):
		return isinstance(object, list) or (isinstance(object,np.ndarray) and object.ndim==1)
