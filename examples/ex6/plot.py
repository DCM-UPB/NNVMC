from pylab import *

data_init = genfromtxt("plot_init_wf.txt")
data_opt = genfromtxt("plot_opt_wf.txt")

figure()

plot(data_init[:,0], data_init[:,1])
plot(data_opt[:,0], data_opt[:,1])
legend(["init", "opt"])

show()
