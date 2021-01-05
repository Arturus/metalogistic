import main
import numpy as np
import matplotlib.pyplot as plt
from timing_decorator import timeit

def prettyPrint(x,y): print("\n".join("{}     {}".format(x, y) for x, y in zip(x, y)))

ps = [.35,.5,.95]
xs = [-5, 2, 20]

@timeit
def createAndPlot():
	m = main.MetaLogistic(ps, xs,lbound=-50)
	ps_from = .0001
	p = np.linspace(ps_from,1-ps_from,100)
	x = m.ppf(p)
	print('a vector:',m.a_vector)
	print("Fit method used:", m.fit_method_used)
	print("Success:",m.success)
	return m,p,x

m,p,x = createAndPlot()

@timeit
def displayPlot():
	fig,ax = plt.subplots()
	ax.plot(x,p)
	ax.scatter(xs, ps, marker='x')
	plt.show()
displayPlot()

@timeit
def pdf():
	x = [1,2,10]
	y = m.pdf(x)
	print("PDF")
	prettyPrint(x, y)

pdf()

@timeit
def cdf():
	x = [1,2,10]
	y = m.cdf(x)
	print("CDF")
	prettyPrint(x,y)

cdf()


@timeit
def quantile():
	p = [.2,.5,.99]
	x = m.ppf(p)
	print("Quantile")
	prettyPrint(p,x)

quantile()

@timeit
def rvs():
	s = m.rvs(size=5)
	print('5 random samples',s)

rvs()