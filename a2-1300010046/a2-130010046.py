import numpy as np
import matplotlib.pyplot as plt
import hope

EPSILON = 2**-52

def generate_mesh(N):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    return X, Y

def setbc(func, x_mesh, y_mesh):
    phi = func(x_mesh, y_mesh)
    phi[1:-1, 1:-1] = 0
    return phi

def func_q(x, y):
    return (x + y)*(x - y)

def phi_actual(N):
    X, Y = generate_mesh(N)
    return func_q(X, Y)


def error_q(phi_new, phi_old):
    return (np.sum((phi_new - phi_old)**2))**0.5/phi_old.shape[0]

def err_res(phi_in):
    return error_q(phi_in, phi_actual(phi_in.shape[0]))


def jacobi_step(phi_in):
    phi_new = phi_in.copy()
    phi_new[1:-1, 1:-1] = 0.25*(phi_in[0:-2, 1:-1] + phi_in[2:,1:-1] + phi_in[1:-1,0:-2] + phi_in[1:-1,2:])
    return phi_new

def gs_step(phi_in):
    phi_next = phi_in.copy()
    N = phi_in.shape[0]
    for i in range(1, N-1):
        for j in range(1, N-1):
            phi_next[i, j] = (phi_in[i+1, j] + phi_in[i, j+1] + phi_next[i-1, j] + phi_next[i, j-1])*0.25

    return phi_next

hope_jacobi_step = hope.jit(jacobi_step)
hope_gs_step = hope.jit(gs_step)
def sor(phi_in, nmax, w):
    N = phi_in.shape[0]
    niter = 0
    error = 1.0
    res = 1.0
    error_sor = np.array([])
    res_sor = np.array([])
    phi_new = phi_in.copy()
    while(error >= 2*EPSILON and niter <= nmax):
        phi_old = phi_new.copy()
        for i in range (1, N-1):
            for j in range (1, N-1):
                phi_new[i,j] = (1 - w)*phi_old[i,j] + w*0.25*(phi_new[i+1,j]+phi_new[i-1,j]+phi_new[i,j+1]+phi_new[i,j-1])
        error = error_q(phi_new, phi_old)
        res = err_res(phi_new)
        error_sor = np.append(error_sor.copy(), error)
        res_sor = np.append(res_sor.copy(), res)
        niter += 1
    return error_sor, res_sor

def solve_lap(phi_in, method, nmax):
    N = phi_in.shape[0]
    error_arr = np.array([])
    res_arr = np.array([])
    niter = 0
    error = 1.0
    res = 1.0
    while(error >= 2*EPSILON and niter <= nmax):
        phi_new = method(phi_in.copy())
        error = error_q(phi_new, phi_in)
        res = err_res(phi_new)
        phi_in = phi_new.copy()
        error_arr = np.append(error_arr.copy(), error)
        res_arr = np.append(res_arr.copy(), res)
        niter += 1
    return error_arr, res_arr

def plotfig(name, title, ydata, xlabel, ylabel, legend, xdata = [], loc = 1):
    plt.figure()
    plt.title(title)
    for i in range(len(ydata)):
        if (xdata == []):
            plt.semilogy(ydata[i], label = legend[i])
        else:
            plt.semilogy(xdata, ydata[i], label = legend[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc = loc)
    plt.savefig(name)
    plt.close()


def ques1():
    for N in [11, 21, 41, 101]:
        x, y = generate_mesh(N)
        phi_bc = setbc(func_q, x, y)
        error1 = solve_lap(phi_bc, jacobi_step, 200000)
        error_jac = error1[0]
        res_jac = error1[1]
        error2 = solve_lap(phi_bc, gs_step, 20000)
        error_gs = error2[0]
        res_gs = error2[1]
        plotfig('N_'+str(N)+'.png', 'Error v/s iteration no. for N='+str(N), [error_jac, error_gs], 'Iteration no.', 'Error (log scale)', ['Jacobi', 'Gauss-Siedel'])
        plotfig('N_'+str(N)+'_1.png', 'Residue v/s iteration no. for N='+str(N), [res_jac, res_gs], 'Iteration no.', 'Residue(log scale)', ['Jacobi', 'Gauss-Siedel'])


def sor_range(phi_in, nmax, wmin, wmax, wstep): 
    N = phi_in.shape[0]
    w = np.arange(wmin, wmax, wstep)
    error = np.zeros(len(w))
    niter = np.zeros(len(w))
    residue = np.zeros(len(w))
    for i in range(len(w)):
        err = sor(phi_in, nmax, w[i])
        error[i] = err[0][-1]
        residue[i] = err[1][-1]
        niter[i] = err[0].shape[0]
    wminerr = w[np.argmin(error)]
    return error, niter, wminerr, residue


def ques2():
    N = 41
    nmax = 20
    x, y = generate_mesh(N)
    phi_bc = setbc(func_q, x, y)
    w = np.arange(0.1, 2, 0.1)
    error, niter, wopt, residue = sor_range(phi_bc, nmax, 0.1, 2, 0.1)
    plotfig('nmax_20.png', 'Error after 20 iterations v/s w', [error], 'w', 'Error', ['SOR error'], w)
    plotfig('nmax_20_1.png', 'Residue after 20 iterations v/s w', [residue], 'w', 'Error', ['SOR error'], w)

    return wopt

def ques3():
    N = 41
    x, y = generate_mesh(N)
    phi_bc = setbc(func_q, x, y)
    w = np.arange(0, 2, 0.1)
    error = np.zeros((len(w), 3))
    niter = np.zeros((len(w), 3))
    residue = np.zeros((len(w), 3))
    wopt = np.zeros(3)
    i = 0
    for nmax in [20, 50, 100]:
        error[:, i], niter[:,i], wopt[i], residue[:,i] = sor_range(phi_bc, nmax, 0, 2, 0.1)
        i += 1
    plotfig('nmaxhigh.png', 'Error v/s w in SOR', [error[:, 0], error[:, 1], error[:, 2]], 'w', 'Error', ['20 iterations', '50 iterations', '100 iterations'], w, 3)
    plotfig('nmaxhigh_1.png', 'Error v/s w in SOR', [residue[:, 0], residue[:, 1], residue[:, 2]], 'w', 'Residue', ['20 iterations', '50 iterations', '100 iterations'], w, 3)
    return wopt

def ques4(wopt):
    N = 41
    x, y = generate_mesh(N)
    phi_bc = setbc(func_q, x, y)
    w = np.arange(wopt - 0.1, wopt + 0.1, 0.01)
    error, niter, w_opt, residue = sor_range(phi_bc, 50, wopt-0.1, wopt + 0.1, 0.01)
    plotfig('optw.png', 'Error v/s w in SOR', [error], 'w', 'Error', ['20 iterations', '50 iterations', '100 iterations'], w)
    plotfig('optw_1.png', 'Residue v/s w in SOR', [error], 'w', 'Error', ['20 iterations', '50 iterations', '100 iterations'], w)
    return w_opt

def ques5():
    N = 101
    nmax = 100
    x, y = generate_mesh(N)
    phi_bc = setbc(func_q, x, y)
    w = np.arange(0.1, 2.1, 0.1)
    error, niter, wopt, residue = sor_range(phi_bc, nmax, 0.1, 2.1, 0.1)
    plotfig('nmax_100_N_101.png', 'Error v/s w', [error], 'w', 'Error', ['SOR error'], w)
    plotfig('nmax_100_N_101_1.png', 'Residue v/s w', [residue], 'w', 'Residue', ['SOR residue'], w)
    w = np.arange(wopt - 0.1, wopt + 0.1, 0.01)
    error, niter, wopt, residue = sor_range(phi_bc, nmax, wopt - 0.1, wopt + 0.1, 0.01)
    plotfig('nmax_100_N_101_2.png', 'Error v/s w', [error], 'w', 'Error', ['SOR error'], w)
    plotfig('nmax_100_N_101_2_1.png', 'Residue v/s w', [residue], 'w', 'Residue', ['SOR residue'], w)
    return wopt

def ques6():
    N = 101
    nmax = 20000
    x, y = generate_mesh(N)
    phi_bc = setbc(func_q, x, y)
    w = np.arange(1.0, 2.0, 0.05)
    error, niter, wopt, residue = sor_range(phi_bc, nmax, 1.0, 2.0, 0.05)
    plotfig('niter_w.png', 'No. of iterations v/s w', [niter], 'w', 'No. of iterations', ['No. of iterations'], w)
    plotfig('niter_w_1.png', 'No. of iterations v/s w', [niter], 'w', 'No. of iterations', ['No. of iterations'], w)
    wopt = w[np.argmin(niter)]
    print wopt
    return wopt

def ques7(wopt):
    N = 101
    x, y = generate_mesh(N)
    phi_bc = setbc(func_q, x, y)
    error_jac = solve_lap(phi_bc, jacobi_step, 100000)[0]
    error_gs = solve_lap(phi_bc, gs_step, 100000)[0]
    error_sor = sor(phi_bc, 100000, wopt)[0]
    plotfig('Comparison.png', 'Convergence of Jacobi, Gauss-Siedel and SOR', [error_jac, error_gs, error_sor], 'Iteration no.', 'Error(log scale)', ['Jacobi', 'Gauss Siedel', 'SOR'])

def main():
    ques1()
    print "Q1 complete"
    ques2()
    print "Q2 complete"
    wopt = ques3()
    print "Q3 complete"
    print wopt
    ques4(wopt[0])
    print "Q4 complete"
    wopt2 = ques5()
    print "Q5 compl"
    ques6()
    print "Q6 compl"
    ques7(wopt2)
    print "Done"

if __name__ =='__main__':
    main()
