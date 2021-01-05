import main
import numpy as np
import matplotlib.pyplot as plt
from timing_decorator import timeit

def prettyPrint(x,y): print("\n".join("{}     {}".format(x, y) for x, y in zip(x, y)))

ps = [.35,.5,.85,.95]
xs = [1, 2, 4, 5]

@timeit
def createAndPlot():
	m = main.MetaLogistic(ps, xs,lbound=-1)
	p = np.linspace(0,1,100)
	x = m.ppf(p)
	print('a:',m.a_vector)
	return m,p,x

m,p,x = createAndPlot()

@timeit
def displayPlot():
	fig,ax = plt.subplots()
	ax.plot(x,p)
	ax.scatter(xs, ps, marker='x')
	plt.show()

@timeit
def pdf():
	x = np.linspace(1,10,10)
	y = m.pdf(x)
	print("PDF")
	prettyPrint(x, y)

pdf()

@timeit
def cdf():
	x = np.linspace(1, 10, 10)
	y = m.cdf(x)
	print("CDF")
	prettyPrint(x,y)

cdf()

@timeit
def rvs():
	s = m.rvs(size=1000)
	print(s[:30],'...')

rvs()