from pylab import *

# since the two ffnn plot data files are for x=0.5 and y=0 respectively
# we can simplify our comparison functions here

def f_v_1(x):
    return exp(-x**2);

def f_d1_1(x):
    return -2.0*x * f_v_1(x)

def f_d2_1(x):
    return -2.0 * f_v_1(x) - 2.0*x * f_d1_1(x)

x0 = 0.5
def f_v_2(y):
    return exp(-x0**2)*exp(-y**2)

def f_d1_2(y):
    return -2.0*y * f_v_2(y)

def f_d2_2(y):
    return -2.0 * f_v_2(y) - 2.0*y * f_d1_2(y)

fnames = ['v_0_0','v_1_0', 'd1_0_0', 'd1_1_0', 'd2_0_0', 'd2_1_0']

files = {}
for fname in fnames:
    files[fname + '_' + 'NN'] = './' + fname + '.txt'

data = {}
for file in files:
    x = []
    nn = []
    for line in open(files[file]):
        llist = line.split('   ')
        x.append(float(llist[0]))
        nn.append(float(llist[1]))
    data[file] = [array(x), array(nn)]

vdata = dict((key, data[key]) for key in data if key.split('_')[0]=='v')
d1data = dict((key, data[key]) for key in data if key.split('_')[0]=='d1')
d2data = dict((key, data[key]) for key in data if key.split('_')[0]=='d2')

gaussv = {}
gaussd1 = {}
gaussd2 = {}

for key in vdata:
    gausl = []
    for x in vdata[key][0]:
        if key.split('_')[1]=='0': gausl.append(f_v_1(x))
        else: gausl.append(f_v_2(x))
    gaussv[key] = array(gausl)

for key in d1data:
    gausl = []
    for x in d1data[key][0]:
        if key.split('_')[1]=='0': gausl.append(f_d1_1(x))
        else: gausl.append(f_d1_2(x))
    gaussd1[key] = array(gausl)

for key in d2data:
    gausl = []
    for x in d2data[key][0]:
        if key.split('_')[1]=='0': gausl.append(f_d2_1(x))
        else: gausl.append(f_d2_2(x))
    gaussd2[key] = array(gausl)

gauss = gaussv.copy()
gauss.update(gaussd1)
gauss.update(gaussd2)

diffs = {}
for key in data:
    diff = 0
    it = 0
    for val in data[key][1]:
        diff += (val-gauss[key][it])**2
        it += 1
    diff /= it
    diffs[key] = sqrt(diff)

diffv = dict((key, diffs[key]) for key in diffs if key.split('_')[0]=='v')
diffd1 = dict((key, diffs[key]) for key in diffs if key.split('_')[0]=='d1')
diffd2 = dict((key, diffs[key]) for key in diffs if key.split('_')[0]=='d2')

figs = {}
axs = {}

fig = figure(1)
ax = fig.add_subplot(111)
ax.plot(vdata[list(vdata.keys())[0]][0], gaussv[list(vdata.keys())[0]], label='Gauss_0')
ax.plot(vdata[list(vdata.keys())[1]][0], gaussv[list(vdata.keys())[1]], label='Gauss_1')
for key in vdata:
    ax.plot(vdata[key][0], vdata[key][1], '--', label=key[2:])
ax.set_ylabel('f(x)')
ax.set_title('Fitted NNs vs. Gaussian')
figs['v'] = fig
axs['v'] = ax

fig = figure(2)
ax = fig.add_subplot(111)
ax.plot(d1data[list(d1data.keys())[0]][0], gaussd1[list(d1data.keys())[0]], label='Gauss_0')
ax.plot(d1data[list(d1data.keys())[1]][0], gaussd1[list(d1data.keys())[1]], label='Gauss_1')
for key in d1data:
    ax.plot(d1data[key][0], d1data[key][1], '--', label=key[3:])
ax.set_ylabel('d/dx f(x)')
ax.set_title('Fitted NNs vs. Gaussian, first derivative')
figs['d1'] = fig
axs['d1'] = ax

fig = figure(3)
ax = fig.add_subplot(111)
ax.plot(d2data[list(d2data.keys())[0]][0], gaussd2[list(d2data.keys())[0]], label='Gauss_0')
ax.plot(d2data[list(d2data.keys())[1]][0], gaussd2[list(d2data.keys())[1]], label='Gauss_1')
for key in d2data:
    ax.plot(d2data[key][0], d2data[key][1], '--', label=key[3:])
ax.set_ylabel('d^2/dx^2 f(x)')
ax.set_title('Fitted NNs vs. Gaussian, second derivative')
figs['d2'] = fig
axs['d2'] = ax

for key in axs:
    axs[key].set_xlabel('x')
    axs[key].set_xlim([-2.5,2.5])
    handles, labels = axs[key].get_legend_handles_labels()
    axs[key].legend(handles, labels, loc='upper right')

# Commented out, because these plots only make sense for multipe NNs
#dfigs = {}
#daxs = {}

#fig = figure(4)
#ax = fig.add_subplot(111)
#x, y = zip(*sorted(diffv.items(), key=lambda x: -x[1]))
#ax.plot(y, '-o')
#ax.set_xticks(arange(len(x)))
#ax.set_xticklabels([z[2:] for z in x])
#ax.set_title('RMSE of Fitted NNs vs. Gaussian')
#dfigs['v'] = fig
#daxs['v'] = ax

#fig = figure(5)
#ax = fig.add_subplot(111)
#x, y = zip(*sorted(diffd1.items(), key=lambda x: -x[1]))
#ax.plot(y, '-o')
#ax.set_xticks(arange(len(x)))
#ax.set_xticklabels([z[3:] for z in x])
#ax.set_title('RMSE of Fitted NNs vs. Gaussian, first derivative')
#dfigs['d1'] = fig
#daxs['d1'] = ax

#fig = figure(6)
#ax = fig.add_subplot(111)
#x, y = zip(*sorted(diffd2.items(), key=lambda x: -x[1]))
#ax.plot(y, '-o')
#ax.set_xticks(arange(len(x)))
#ax.set_xticklabels([z[3:] for z in x])
#ax.set_title('RMSE of Fitted NNs vs. Gaussian, second derivative')
#dfigs['d2'] = fig
#daxs['d2'] = ax

#for key in daxs:
#    daxs[key].set_ylabel('RMSE')


#for key in dfigs:
#    dfigs[key].savefig('diff_'+key+'.pdf')

#for key in figs:
#    figs[key].savefig('comp_'+key+'.pdf')

show()
