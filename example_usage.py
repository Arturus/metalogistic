import main
import numpy as np
import matplotlib.pyplot as plt
from timing_decorator import timeit

ps = [.35,.5,.85,.95]
xs = [1, 2, 4, 5]

@timeit
def do():
	m = main.MetaLogistic(ps, xs)
	p = np.linspace(0,1,100)
	x = m.ppf(p)
	print('a:',m.a_vector)
	return p,x

p,x = do()
fig,ax = plt.subplots()
ax.plot(x,p)
ax.scatter(xs, ps, marker='x')
plt.show()
