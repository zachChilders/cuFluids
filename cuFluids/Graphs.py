import numpy as np
import scipy.special
from bokeh.plotting import *

mu, sigma = 0, 0.5

#measured = np.random.normal(mu, sigma, 1000)
file = open("C:\Users\Zach\Documents\Visual Studio 2012\Projects\cuFluids\Solver2D\Solver2D\log.txt", "r")
measured = []
for line in file:
    fl = float(line)
    measured.append(fl)
file.close()

mean = (max(measured) + min(measured)) / len(measured)

centered = map(lambda x: x - mean, measured)

hist, edges = np.histogram(centered, density=True, bins=100)

x = np.linspace(-2, 2, 1000)
pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))
cdf = (1+scipy.special.erf((x-mu)/np.sqrt(2*sigma**2)))/2

output_file('histogram.html')

hold()

figure(title="Vector Magnitudes",tools="previewsave",
       background_fill="#E8DDCB")
quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
     fill_color="#036564", line_color="#033649",
)

# Use `line` renderers to display the PDF and CDF
#line(x, pdf, line_color="#D95B43", line_width=8, alpha=0.7, legend="PDF")
#line(x, cdf, line_color="white", line_width=2, alpha=0.7, legend="CDF")

legend().orientation = "top_left"

xax, yax = axis()
xax.axis_label = 'X Force'
yax.axis_label = 'Magnitudes'

show()