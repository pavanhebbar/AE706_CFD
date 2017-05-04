import numpy as np
import matplotlib.pyplot as plt

def function1(x):
	return (1 - np.cos(x))/x**2

def question():
	x = np.linspace(-4*10**-8, 4*10**-8, 10000)
	y = function1(x)
	plt.figure()
	plt.title("Plot function")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.plot(x, y)
	plt.show()

question()