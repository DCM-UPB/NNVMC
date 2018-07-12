from pylab import *

data = genfromtxt("plot_opt_wf.txt")

figure()

plot(data[:,0], data[:,1])

show()
