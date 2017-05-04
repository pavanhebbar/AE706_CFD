import numpy as np
import matplotlib.pyplot as plt

def newrap1(c): 
	c = float(c)
	t = c
	while (t*t - c > 0.0):
		t = (c/t + t)/2.0
	return t

def newrap2(c):
	c = float(c)
	EPSILON = 1e-15
	t = c
	while (abs(t*t - c) > EPSILON):
		t = (c/t + t)/2.0
		print t
	return t

def newrap3(c):
	c = float(c)
	EPSILON = 1e-15
	t = c
	while (abs(t*t - c) > EPSILON*c):
		t = (c/t + t)/2.0
		print t
	return t