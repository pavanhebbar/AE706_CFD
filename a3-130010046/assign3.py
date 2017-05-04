import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def del_t(cfl, delx, a):
    return abs(cfl*delx/a)

def ftcs(cfl, u_in, periodic = 0):
    u_next = u_in.copy()
    u_next[1: -1] = u_in[1: -1] - cfl/2*(u_in[2:] - u_in[0:-2])
    if (periodic == 0):
        u_temp = 2*u_in[-1] - u_in[-2]
        u_next[-1] = u_in[-1] - cfl/2*(u_temp - u_in[-2])
    else:
        u_next[-1] = u_in[-1] - cfl/2*(u_in[1] - u_in[-2])
        u_next[0] = u_next[-1]
    return u_next

def ftfs(cfl, u_in, periodic = 0):
    u_next = u_in.copy()
    u_next[1: -1] = u_in[1: -1] - cfl*(u_in[2:] - u_in[1:-1])
    if (periodic == 0):
        u_temp = 2*u_in[-1] - u_in[-2]
        u_next[-1] = u_in[-1] - cfl*(u_temp - u_in[-1])
    else:
        u_next[-1] = u_in[-1] - cfl*(u_in[1] - u_in[-1])
        u_next[0] = u_next[-1]
    return u_next

def ftbs(cfl, u_in, periodic = 0):
    u_next = u_in.copy()
    u_next[1:] = u_in[1:] - cfl*(u_in[1:] - u_in[0:-1])
    if (periodic == 1):
        u_next[0] = u_next[-1]
    return u_next

def ftcs2(cfl, u_in, periodic = 0):
    u_next = u_in.copy()
    u_next[1:-1] = u_in[1: -1] - cfl/2*(u_in[2:] - u_in[0:-2]) + cfl**2/2*(u_in[2:] - 2*u_in[1:-1] + u_in[0:-2])
    if periodic == 0:
        u_temp = 2*u_in[-1] - u_in[-2]
        u_next[-1] = u_in[-1] - cfl/2*(u_temp - u_in[-2]) + cfl**2/2*(u_temp - 2*u_in[-1] + u_in[-2])
    else:
        u_next[-1] = u_in[-1] - cfl/2*(u_in[1] - u_in[-2]) + cfl**2/2*(u_in[1] - 2*u_in[-1] + u_in[-2])
        u_next[0] = u_next[-1]
    return u_next

def solve(method, xrange, u_in, tmax, cfl, a, periodic = 0):
    nx = u_in.shape[0] - 1
    delx = float (xrange/nx)
    delt = del_t(cfl, delx, a)
    nt = int (tmax/delt)
    u_matrix = np.zeros((nt + 1, nx + 1))
    u_matrix[0, :] = u_in.copy()
    for i in range(1, nt + 1):
        u_matrix[i, :] = method(cfl, u_matrix[i - 1, :], periodic)
    return u_matrix

def pltimage(name, title, xlabel, ylabel, xbound, ybound, imgarray, xlbound = 0, ylbound = 0):
    plt.figure()
    plt.title(title)
    plt.imshow(imgarray, aspect = 'auto', extent = [xlbound, xbound, ylbound, ybound], origin = 'low')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.savefig(name)
    plt.close()

def pltfig(name, title, xlabel, ylabel, xdata, ydata, legend, log = 0):
    plt.figure()
    plt.title(title)
    for i in range(len(ydata)):
        if (xdata[i] == []):
            plt.plot(ydata[i], label = legend[i])
        else:
            plt.plot(xdata[i], ydata[i], label = legend[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if(log == 1):
        plt.yscale('log')
    plt.legend(loc = 1)
    plt.savefig(name)
    plt.close()

def plot_q1q2(method, u_in, index, mthd_str, xmax = 1.0):
    x = np.linspace(0, xmax, u_in.shape[0])
    u_sol1 = solve(method, xmax, u_in, 1.2, 0.8, 1.0)
    u_sol2 = solve(method, xmax, u_in, 1.2, 1.0, 1.0)
    u_sol3 = solve(method, xmax, u_in, 1.2, 1.2, 1.0)
    pltimage(mthd_str + str(index) + "_08.png", mthd_str + " scheme with CFL = 0.8", 'x', 't', 2.0, 1.2, u_sol1)
    pltimage(mthd_str + str(index) + "_1.png", mthd_str + " scheme with CFL = 1.0", 'x', 't', 2.0, 1.2, u_sol2)
    pltimage(mthd_str + str(index) + "_12.png", mthd_str + " scheme with CFL = 1.2", 'x', 't', 2.0, 1.2, u_sol3)
    n08 = u_sol1.shape[0]
    n1 = u_sol2.shape[0]
    n12 = u_sol3.shape[0]
    pltfig(mthd_str + str(index) + "_08_1.png", mthd_str + " scheme with CFL = 0.8", "x", "u", [x, x, x], [u_sol1[int(n08/4), :], u_sol1[int(n08/2), :], u_sol1[int(3*n08/4), :]], ["t = 0.3s", "t = 0.6s", "t = 0.9s"])
    pltfig(mthd_str + str(index) + "_08_1_log.png", mthd_str + " scheme with CFL = 0.8", "x", "u", [x, x, x], [u_sol1[int(n08/4), :], u_sol1[int(n08/2), :], u_sol1[int(3*n08/4), :]], ["t = 0.3s", "t = 0.6s", "t = 0.9s"], 1)
    pltfig(mthd_str + str(index) + "_1_1.png", mthd_str + " scheme with CFL = 1.0", "x", "u", [x, x, x], [u_sol2[int(n1/4), :], u_sol2[int(n1/2), :], u_sol2[int(3*n1/4), :]], ["t = 0.3s", "t = 0.6s", "t = 0.9s"])
    pltfig(mthd_str + str(index) + "_1_1_log.png", mthd_str + " scheme with CFL = 1.0", "x", "u", [x, x, x], [u_sol2[int(n1/4), :], u_sol2[int(n1/2), :], u_sol2[int(3*n1/4), :]], ["t = 0.3s", "t = 0.6s", "t = 0.9s"], 1)
    pltfig(mthd_str + str(index) + "_12_1.png", mthd_str + " scheme with CFL = 1.2", "x", "u", [x, x, x], [u_sol1[int(n12/4), :], u_sol3[int(n12/2), :], u_sol3[int(3*n12/4), :]], ["t = 0.3s", "t = 0.6s", "t = 0.9s"])
    pltfig(mthd_str + str(index) + "_12_1_log.png", mthd_str + " scheme with CFL = 1.2", "x", "u", [x, x, x], [u_sol1[int(n12/4), :], u_sol3[int(n12/2), :], u_sol3[int(3*n12/4), :]], ["t = 0.3s", "t = 0.6s", "t = 0.9s"], 1)


def ques1():
    nx = 50
    u_in = np.zeros(nx + 1, dtype = float)
    x = np.linspace(0, 1, 51)
    u_in[0] = 1.0
    plot_q1q2(ftbs, u_in, 1, "FTBS")
    plot_q1q2(ftcs, u_in, 1, "FTCS")
    plot_q1q2(ftfs, u_in, 1, "FTFS")    



def ques2():
    nx = 100
    x = np.linspace(0, 3, 3*nx + 1)
    u_in = np.sin(2*np.pi*x)
    u_in[0] = 0
    u_in[nx+1:] = 0
    plot_q1q2(ftbs, u_in, 2, "FTBS", 3.0)
    plot_q1q2(ftcs, u_in, 2, "FTCS", 3.0)
    plot_q1q2(ftfs, u_in, 2, "FTFS", 3.0)
    u_in = np.sin(2*np.pi*x) + np.sin(20*np.pi*x)
    u_in[nx+1:] = 0
    plot_q1q2(ftbs, u_in, 3, "FTBS", 3.0)    
    plot_q1q2(ftcs, u_in, 3, "FTCS", 3.0)
    plot_q1q2(ftfs, u_in, 3, "FTFS", 3.0)


def q3_t1():
    nx = 40
    x = np.linspace(-1, 1, nx + 1)
    u_in = -np.sin(np.pi*x) 
    u_in[0] = 0
    u_bs = solve(ftbs, 2.0, u_in, 30, 0.8, 1.0, 1)
    u_cs2 = solve(ftcs2, 2.0, u_in, 30, 0.8, 1.0, 1)
    nt = u_bs.shape[0]
    pltimage("laney_t1_bs.png", "Test case 1 using FTBS with CFL = 0.8", 'x', 't', 1, 30, u_bs, -1)
    pltimage("laney_t1_cs2.png", "Test case 1 using FTCS2 with CFL = 0.8", 'x', 't', 1, 30, u_cs2, -1)
    pltfig("laney_t1_bs_1.png", "Test case 1 using FTBS with CFL = 0.8", 'x', 'u', [x, x], [u_bs[0], u_bs[nt - 1]], ["t = 0s", "t = 30s"])
    pltfig("laney_t1_cs2_1.png", "Test case 1 using FTCS2 with CFL = 0.8", 'x', 'u', [x, x], [u_cs2[0], u_cs2[nt -1]], ["t = 0s", "t = 30s"])

def q3_t2():
    nx = 40
    x = np.linspace(-1.0, 1.0, 1*nx + 1)
    u_in = np.zeros(1*nx + 1)
    for i in range(len(x)):
        if (abs(x[i]) < float (1.0/3.0)):
            u_in[i] = 1.0
        else:
            u_in[i] = 0.0
    u_bs = solve(ftbs, 2.0, u_in, 4, 0.8, 1.0, 1)
    u_cs2 = solve(ftcs2, 2.0, u_in, 4, 0.8, 1.0, 1)
    nt = u_bs.shape[0]
    pltimage("laney_t2_bs.png", "Test case 2 using FTBS with CFL = 0.8", 'x', 't', 1, 4, u_bs, -1)
    pltimage("laney_t2_cs2.png", "Test case 2 using FTCS2 with CFL = 0.8", 'x', 't', 1, 4, u_cs2, -1)
    pltfig("laney_t2_bs_1.png", "Test case 2 using FTBS with CFL = 0.8", 'x', 'u', [x, x], [u_bs[0], u_bs[nt - 1]], ["t = 0s", "t = 4s"])
    pltfig("laney_t2_cs2_1.png", "Test case 2 using FTCS2 with CFL = 0.8", 'x', 'u', [x, x], [u_cs2[0], u_cs2[nt -1]], ["t = 0s", "t = 4s"])

def q3_t3():
    nx = 600
    x = np.linspace(-1.0, 1.0, 1*nx + 1)
    u_in = np.zeros(1*nx + 1)
    for i in range(len(x)):
        if (abs(x[i]) < float (1.0/3.0)):
            u_in[i] = 1.0
        else:
            u_in[i] = 0.0
    u_bs = solve(ftbs, 2.0, u_in, 40, 0.8, 1.0, 1)
    u_cs2 = solve(ftcs2, 2.0, u_in, 40, 0.8, 1.0, 1)
    nt = u_bs.shape[0]
    pltimage("laney_t3_bs.png", "Test case 2 using FTBS with CFL = 0.8", 'x', 't', 1, 40, u_bs, -1)
    pltimage("laney_t3_cs2.png", "Test case 2 using FTCS2 with CFL = 0.8", 'x', 't', 1, 40, u_cs2, -1)
    pltfig("laney_t3_bs_1.png", "Test case 2 using FTBS with CFL = 0.8", 'x', 'u', [x, x, x], [u_bs[0], u_bs[int(nt/10)], u_bs[nt - 1]], ["t = 0s", "t = 4s", "t = 40s"])
    pltfig("laney_t3_cs2_1.png", "Test case 2 using FTCS2 with CFL = 0.8", 'x', 'u', [x, x, x], [u_cs2[0], u_cs2[int(nt/10)], u_cs2[nt -1]], ["t = 0s", "t = 4s", "t = 40s"])

def ques3():
    q3_t1()
    q3_t2()
    q3_t3()

def main():
    ques1()
    ques2()
    ques3()


if __name__=='__main__':
    main()