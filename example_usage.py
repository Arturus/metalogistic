from metalogistic.main import MetaLogistic

# Example 1
ps = [.15, .5, .9]
xs = [-20, -1, 40]

m = MetaLogistic(ps, xs)

m.print_summary()
m.display_plot()

cdf_probabilities = m.cdf(10)
print("cdf() demo:", cdf_probabilities)

pdf_densities = m.pdf([10, 20, 21])
print("pdf() demo:", pdf_densities)

quantiles = m.quantile([0.8, .99])
print("quantile() demo:", quantiles)

# Example 2
print('\n')

ps = [.1, .5, .9]
xs = [-20, -1, 120]

m = MetaLogistic(ps, xs)

m.print_summary()
m.display_plot(hide_extreme_densities=10)

# Example 3
print('\n')

ps = [.07,.15, .5, .9]
xs = [-35,-20, -1, 50]

m = MetaLogistic(ps, xs, ubound=70)

m.print_summary()
m.display_plot(x_from_to=(-100, 60))