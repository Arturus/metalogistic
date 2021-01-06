from metalogistic.main import MetaLogistic

ps = [.15, .5, .9]
xs = [-20, -1, 40]

m = MetaLogistic(ps, xs, ubound=100)

m.printSummary()
m.displayPlot()

cdf_probabilities = m.cdf(10)
print("cdf() demo:", cdf_probabilities)

pdf_densities = m.pdf([10, 20, 21])
print("pdf() demo:", pdf_densities)

quantiles = m.quantile([0.8, .99])
print("quantile() demo:", quantiles)
