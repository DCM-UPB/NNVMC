from pylab import *

data_r1 = genfromtxt("plot_opt_wf_r1.txt")
data_r2 = genfromtxt("plot_opt_wf_r2.txt")

figure()

plot(data_r1[:,0], data_r1[:,1])
plot(data_r2[:,0], data_r2[:,1])

show()
