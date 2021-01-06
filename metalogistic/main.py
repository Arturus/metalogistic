import numpy as np
from scipy import optimize
from scipy import stats
import matplotlib.pyplot as plt

class MetaLogistic(stats.rv_continuous):
	'''
	We subclass scipy.stats.rv_continuous so we can make use of all the nice SciPy methods. We redefine the private methods
	_cdf, _pdf, and _ppf, and SciPy will make calls to these whenever needed.
	'''
	def __init__(self, cdf_ps=None, cdf_xs=None, term=None, fit_method=None, lbound=None, ubound=None, a_vector=None):
		'''
		:param cdf_ps: Probabilities of the CDF input data.
		:param cdf_xs: X-values of the CDF input data (the pre-images of the probabilities).
		:param term: Produces a `term`-term metalog. Cannot be greater than the number of CDF points provided. By default, it is
		equal to the number of CDF points.
		:param fit_method: Set to 'LLS' to allow linear least squares only. By default, numerical methods are tried if LLS fails.
		:param lbound: Lower bound
		:param ubound: Upper bound
		:param a_vector: You may supply the a-vector directly, in which case the input data `cdf_ps` and `cdf_xs` are not used for fitting.
		'''
		super(MetaLogistic, self).__init__()

		if lbound is None and ubound is None:
			self.boundedness = False
		if lbound is None and ubound is not None:
			self.boundedness = 'upper'
		if lbound is not None and ubound is None:
			self.boundedness = 'lower'
		if lbound is not None and ubound is not None:
			self.boundedness = 'bounded'
		self.lbound = lbound
		self.ubound = ubound

		self.fit_method_requested = fit_method
		self.numeric_ls_solver_used = None

		self.cdf_ps = cdf_ps
		self.cdf_xs = cdf_xs
		if cdf_xs is not None and cdf_ps is not None:
			self.cdf_p_x_mapping = {cdf_ps[i]: cdf_xs[i] for i in range(len(cdf_ps))}
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

		if len(cdf_ps) != len(cdf_ps):
			raise ValueError("cdf_ps and cdf_xs must have the same length")

		if term is None:
			self.term = self.cdf_len
		else:
			self.term = term


		self.constructZVec()
		self.constructYMatrix()

		#  Try linear least squares
		self.fitLinearLeastSquares()

		# If linear least squares result is feasible
		if MetaLogistic.aVecFeasibility(self.a_vector) == 0:
			self.fit_method_used = 'LLS'
			self.success = True

		#  If linear least squares result is not feasible
		else:
			# if the user allows it, use numerical least squares
			if not fit_method == 'LLS':
				self.fit_method_used = 'numeric'
				self.fitNumericLeastSquares()
				self.success = self.numeric_ls_OptimizeResult.success

			# If only LLS is allowed, we cannot find a valid metalog
			else:
				self.success = False





	def fitLinearLeastSquares(self):
		'''
		Constructs the a-vector by linear least squares, as defined in Keelin 2016, Equation 7 (unbounded case), Equation 12 (semi-bounded case).

		'''
		left = np.linalg.inv(np.dot(self.YMatrix.T, self.YMatrix))
		right = np.dot(self.YMatrix.T, self.z_vec)

		self.a_vector = np.dot(left, right)

	def fitNumericLeastSquares(self):
		def loss_function(a_candidate):
			bounds_kwargs = {}
			if self.lbound is not None:
				bounds_kwargs['lbound'] = self.lbound
			if self.ubound is not None:
				bounds_kwargs['ubound'] = self.ubound

			# Setting a_vector in this MetaLogistic call overrides the cdf_ps and cdf_xs arguments, which are only used
			# for meanSquareError().
			mlog_candidate = MetaLogistic(self.cdf_ps, self.cdf_xs, **bounds_kwargs, a_vector=a_candidate)

			return mlog_candidate.meanSquareError()

		def infeasibility_score(a_candidate):
			return MetaLogistic.aVecFeasibility(a_candidate)

		feasibility_constraint = optimize.NonlinearConstraint(infeasibility_score, 0, 0)
		optimize_kwargs = {}
		# First, try the SLSQP solver, which is often fast and accurate
		optimize_results = optimize.minimize(loss_function,
											 self.a_vector,
											 constraints=feasibility_constraint,
											 method='SLSQP')
		self.numeric_ls_solver_used = 'SLSQP'

		# If the mean square error is too large, try the trust-constr solver
		if optimize_results.fun > 0.001:
			options = {'xtol':1e-6}
			optimize_results = optimize.minimize(loss_function,
												 self.a_vector,
												 constraints=feasibility_constraint,
												 method='trust-constr',
												 options=options)
			self.numeric_ls_solver_used = 'trust-constr'

		self.a_vector = optimize_results.x

		self.numeric_ls_OptimizeResult = optimize_results

	def meanSquareError(self):
		ps_on_fitted_cdf = self.cdf(self.cdf_xs)
		sum_sq_error = np.sum((self.cdf_ps - ps_on_fitted_cdf) ** 2)
		return sum_sq_error/self.cdf_len

	@staticmethod
	def aVecFeasibility(a_vector):
		'''
		Closed-form feasibility condition on the a-vector, as defined in Keelin 2016, Equation 5.
		Equation 5 is the derivative of Equation 6 with respect to y.
		'''

		# `self.a_vector` is 0-indexed, while in Keelin 2016 the a-vector is 1-indexed.
		# To make this method as easy as possible to read if following along with the paper, I create a dictionary `a`
		# that mimics a 1-indexed vector.
		a = {i + 1: element for i, element in enumerate(a_vector)}

		def series(y):
			# TODO: verify that boundedness does not affect the feasibility condition
			# term 2
			series = a[2] / (y * (1 - y))

			for n in range(3,len(a_vector)+1):
				# term 3
				if n == 3:
					series += a[n]*((y-.5)/(y*(1-y)) + np.log(y/(1-y)))

				if n % 2 == 1 and n >=5:
					'''
					Odd terms greater than or equal to 5 in the M_n series (Eqn. 6), and their derivative w.r.t. y.
					For calculations, see calculations.wxmx.
					You can paste the below into a LaTeX editor.
					
					term = a_n(y-0.5)^{\frac{n-1}{2}}  \\
					\frac{\partial term}{\partial y} = 0.5 a_n (n - 1) (y - 0.5)^{\frac{n - 3}{2}}
					'''
					series += 0.5*a[n]*(n-1)*(y-0.5)**((n-3)/2)

				if n % 2 == 0 and n >=6:
					'''
					Even terms greater than or equal to 6 in the M_n series (Eqn. 6), and their derivative w.r.t. y. 
					For calculations, see calculations.wxmx.
					You can paste the below into a LaTeX editor.
					
					term = a_n(y-0.5)^{\frac{n}{2}-1} \ln\frac{y}{1-y}  \\
					\frac{\partial term}{\partial y} =  a_n\, \left( \frac{n}{2}-1\right) \, {{\left( y-0.5\right) }^{\frac{n}{2}-2}}
					\log{\left( \frac{y}{1-y}\right) }+\frac{a_n\, \left( 1-y\right) \,
					 {{\left( y-0.5\right) }^{\frac{n}{2}-1}}\, \left( \frac{y}{{{\left( 1-y\right) }^{2}}}+\frac{1}{1-y}\right) }{y}
					'''
					series+= a[n] * (n / 2-1) * (y-0.5)**(n / 2-2) * np.log(y / (1-y)) + (a[n] * (1-y) * (y-0.5)**(n / 2-1) * (y / (1-y)**2 + 1 / (1-y))) / y

			return series

		check_ys_from = 0.0001
		ys_to_check = np.linspace(check_ys_from,1-check_ys_from,200)
		series_to_check = series(ys_to_check)
		infeasibilty_score = sum(series_to_check[series_to_check<=0])
		return infeasibilty_score

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
			self.z_vec = -np.log(self.ubound - self.cdf_xs)
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

	def _quantile(self, probability, force_unbounded=False):
		'''
		The metalog inverse CDF, or quantile function, as defined in Keelin 2016, Equation 6 (unbounded case), Equation 11 (semi-bounded case),
		and Equation 14 (bounded case).

		`probability` must be a scalar.
		'''

		if not 0 <= probability <= 1:
			raise ValueError("Probability in call to quantile() must be between 0 and 1")

		if probability == 0:
			if (self.boundedness == 'lower' or self.boundedness == 'bounded') and not force_unbounded:
				return self.lbound
			else:
				return -np.inf

		if probability == 1:
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

	def densitySmallM(self,cumulative_prob):
		'''
		This is the metalog PDF as a function of cumulative probability, as defined in Keelin 2016, Equation 9 (unbounded case),
		Equation 13 (semi-bounded case),
		Notice the unusual definition of the PDF, which is why I call this function densitySmallM in reference to the notation in
		Keelin 2016.
		'''

		if self.isListLike(cumulative_prob):
			return [self.densitySmallM(i) for i in cumulative_prob]

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
		density_functions[3] = 1/(1/density_functions[2] + a[3]*(p05_term/p1p_term+ln_p_term))
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
				density_function = density_function * np.exp(-self._quantile(cumulative_prob, force_unbounded=True))
			elif cumulative_prob == 0:
				density_function = 0
			else:
				raise ValueError("Probability in call to densitySmallM() cannot be equal to 1 with a lower-bounded distribution.")

		if self.boundedness == 'upper':
			if 0 < cumulative_prob < 1:
				density_function = density_function * np.exp(self._quantile(cumulative_prob, force_unbounded=True))
			elif cumulative_prob == 1:
				density_function = 0
			else:
				raise ValueError("Probability in call to densitySmallM() cannot be equal to 0 with a upper-bounded distribution.")

		if self.boundedness == 'bounded':  # Equation 15
			if 0 < cumulative_prob < 1:
				x_unbounded = np.exp(self._quantile(cumulative_prob, force_unbounded=True))
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
		f_to_zero = lambda probability: self._quantile(probability) - x
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
			return self._quantile(probability)

	def quantile(self, probability):
		'''
		An alias for ppf, because 'percent point function' is somewhat non-standard terminology
		'''
		return self._ppf(probability)

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

	def printSummary(self):
		print("Fit method requested:", self.fit_method_requested)
		print("Fit method used:", self.fit_method_used)
		print("Success:", self.success)
		if not self.fit_method_used == 'LLS':
			print("Solver for numeric fit:", self.numeric_ls_solver_used, )
			print("Solver message:", self.numeric_ls_OptimizeResult.message)
		print("Mean square error:", self.meanSquareError())
		print('a vector:', self.a_vector)

	def createCDFPlotData(self,p_from=0.0001,p_to=None):
		if p_to is None:
			p_to = 1-p_from

		cdf_ps = np.linspace(p_from,p_to,200)
		cdf_xs = self.quantile(cdf_ps)

		return {'X-values':cdf_xs,'Probabilities':cdf_ps}

	def createPDFPlotData(self, p_from=0.001, p_to=None):
		if p_to is None:
			p_to = 1 - p_from

		pdf_ps = np.linspace(p_from, p_to, 200)
		pdf_xs = self.quantile(pdf_ps)
		pdf_densities = self.densitySmallM(pdf_ps)

		return {'X-values': pdf_xs, 'Densities': pdf_densities}

	def displayPlot(self, p_from=0.001, p_to=None):
		fig, (cdf_axis, pdf_axis) = plt.subplots(2)

		cdf_data = self.createCDFPlotData(p_from,p_to)
		cdf_axis.plot(cdf_data['X-values'],cdf_data['Probabilities'])
		if self.cdf_xs is not None and self.cdf_ps is not None:
			cdf_axis.scatter(self.cdf_xs, self.cdf_ps, marker='+', color='red')
		cdf_axis.set_title('CDF')

		pdf_data = self.createPDFPlotData(p_from,p_to)
		pdf_axis.plot(pdf_data['X-values'], pdf_data['Densities'])
		pdf_axis.set_title('PDF')

		fig.show()
		return fig

