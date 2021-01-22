import time
from metalogistic.main import MetaLogistic

def timeit(method):
	def timed(*args, **kw):
		ts = time.time()
		result = method(*args, **kw)
		te = time.time()
		if 'log_time' in kw:
			name = kw.get('log_name', method.__name__.upper())
			kw['log_time'][name] = int((te - ts) * 1000)
		else:
			print('%r  %2.2f ms' % \
				  (method.__name__, (te - ts) * 1000))
		return result
	return timed



@timeit
def doFit(cdf_ps, cdf_xs, lbound=None, ubound=None):
	mlog_object = MetaLogistic(cdf_ps,cdf_xs,lbound=lbound,ubound=ubound)
	return mlog_object

@timeit
def createPlotData(mlog_oject):
	mlog_oject.create_cdf_plot_data()
	mlog_oject.create_pdf_plot_data()



def doAll(cdf_ps, cdf_xs, lbound=None, ubound=None):
	print("\n#### Speed test ####")
	print("Data:")
	print("cdf_ps",cdf_ps)
	print("cdf_xs", cdf_xs)
	print("Bounds:",lbound,ubound)
	print("\nTimings:")
	m = doFit(cdf_ps,cdf_xs,lbound,ubound)
	createPlotData(m)
	print("\nSummary printout:")
	m.print_summary()

ps = [.15, .5, .9]
xs = [-20, -1, 40]
doAll(ps,xs)
doAll(ps,xs,lbound=-50)
doAll(ps,xs,ubound=100)
doAll(ps,xs,lbound=-100,ubound=50)

ps = [.15, .5, .9]
xs = [-20, -1, 100]
doAll(ps,xs)
doAll(ps,xs,ubound=1000)
doAll(ps,xs,lbound=-50,ubound=1000)
