import numpy as np 
import matplotlib.pyplot as plt

def machineeps():              #find epsilon of machine
	epsilon = 1.0 
	while (1 + epsilon > 1.0):
		epsilon = epsilon*0.5
	return epsilon

def ques3_func(x):
	return (1 - np.cos(x))/(x*x)

def q3_func_cor1(x):
	return 1/(1 + np.cos(x))*(np.sin(x)/x)**2

def q3_func_cor2(x):
	return 2*(np.sin(x/2)/x)**2

def q3():
	x = np.linspace(-4*1e-8, 4*1e-8, 100)
	f1 = ques3_func(x)
	f2 = q3_func_cor1(x)
	f3 = q3_func_cor2(x)
	plotfig("Q3.png", r'Plots of $(1-cos(x))/(x^2)$ using different forms', x, [f1, f2, f3], 'x', r'$(1-cos(x))/(x^2)$', [r'$\frac{1-cos(x)}{x*x}$', r"$\frac{1}{1 + cos(x)}\left(\frac{sin(x)}{x}\right)^2$", r"$2\left(\frac{sin(0.5x)}{x}\right)^{2}$"])

def q5_func(x):
	return np.sin(x)

def forwarddiff(x, h):
	return (q5_func(x+h) - q5_func(x))/h

def backdiff(x, h):
	return (q5_func(x) - q5_func(x - h))/h

def centraldiff(x, h):
	return (q5_func(x + h) - q5_func(x - h))/(2*h)

def diff_o4(x, h):
	#return (-q5_func(x + 2*h) + 8*q5_func(x + h) - 8*q5_func(x - h) + q5_func(x - 2*h))/(12*h)
	return (4*centraldiff(x, h) - centraldiff(x, 2*h))/3


def plotfig(name, title, x, y, xlabel, ylabel, label, loglog = 0, loc = 1):
	plt.figure()
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	for i in range(len(y)):
		if (loglog == 0):
			plt.plot(x, y[i], label = label[i])
		else:
			plt.loglog(x, y[i], label = label[i])
	plt.legend(loc = loc)
	plt.savefig(name)
	plt.close()

def q5():
	x = np.pi*0.25
	h = np.pi*0.25
	h_array = np.array([h])
	while (h > 1e-10):
		h = h*0.5
		h_array = np.append(h_array.copy(), h)
	
	y_for_err = abs(forwarddiff(x, h_array) - np.cos(np.pi/4))
	y_bac_err = abs(backdiff(x, h_array) - np.cos(np.pi/4))
	y_cen_err = abs(centraldiff(x, h_array) - np.cos(np.pi/4))
	y_o4_err = abs(diff_o4(x, h_array) - np.cos(np.pi/4))
	plotfig("Q5.png", 'Plots of errors in d/dx (sin(x)) v/s h', h_array, [y_for_err, y_bac_err, y_cen_err, y_o4_err], 'h', 'Error', ['Forward difference', 'Backward difference', 'Central difference', 'Fourth order approx'], 1, 2)

def main():
	epsilon = machineeps();
	print "Machine epsilon = ", epsilon
	q3()
	q5()

if __name__=='__main__':
	main()