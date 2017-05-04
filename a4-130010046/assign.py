import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as anim

GAMMA = 1.4
R = 287.1

#q, q_t are 3*N matrices

def diff_1(q_t, start, end, h):         #end column is not included
    e1 = q_t[:, (start + 1):(end + 1)]
    e_1 = q_t[:, (start - 1): (end - 1)]
    return 0.5*(e1 - e_1)/h

def diff_2(q_t, start, end, h):        
    e0 = q_t[:, (start):(end)]
    e1 = q_t[:, (start + 1):(end + 1)]
    e_1 = q_t[:, (start - 1):(end - 1)]
    return (e1 - 2*e0 + e_1)/h**2

def diff_4_cen(q_t, start, end, h):
    e2 = q_t[:, (start + 2):(end + 2)]
    e1 = q_t[:, (start + 1):(end + 1)]
    e0 = q_t[:, (start):(end)]
    e_1 = q_t[:, (start - 1):(end - 1)]
    e_2 = q_t[:, (start - 2):(end - 2)]
    return (e2 - 4*e1 + 6*e0 - 4*e_1 + e_2)/(h**4)

def diff_4_for(q_t, start, end, h):
    e0 = q_t[:, (start):(end)]
    e1 = q_t[:, (start + 1):(end + 1)]
    e2 = q_t[:, (start + 2):(end + 2)]
    e3 = q_t[:, (start + 3):(end + 3)]
    e4 = q_t[:, (start + 4):(end + 4)]
    e5 = q_t[:, (start + 5):(end + 5)]
    return (3*e0 - 14*e1 + 26*e2 - 24*e3 + 11*e4 - 2*e5)/(h**4)

def diff_4_bac(q_t, start, end, h):
    e0 = q_t[:, (start):(end)]
    e1 = q_t[:, (start - 1):(end - 1)]
    e2 = q_t[:, (start - 2):(end - 2)]
    e3 = q_t[:, (start - 3):(end - 3)]
    e4 = q_t[:, (start - 4):(end - 4)]
    e5 = q_t[:, (start - 5):(end - 5)]
    return (3*e0 - 14*e1 + 26*e2 - 24*e3 + 11*e4 - 2*e5)/(h**4)

def getq(q_t):    
    q_1 = q_t[0]
    q_2 = q_t[0]*q_t[1]
    q_3 = q_t[2]/(GAMMA - 1) + q_t[0]*q_t[1]*q_t[1]/2
    return np.vstack([q_1, q_2, q_3])

def getq_tilda(q):
    rho = q[0]
    u = q[1]/q[0]
    p = (q[2] - rho*u*u/2)*(GAMMA - 1)
    return np.vstack([rho, u, p])

def gete(q_t):
    e_1 = q_t[0]*q_t[1]
    e_2 = q_t[0]*q_t[1]*q_t[1] + q_t[2]
    e_3 = q_t[1]*(GAMMA/(GAMMA - 1)*q_t[2] + q_t[0]*q_t[1]*q_t[1]/2)
    return np.vstack([e_1, e_2, e_3])

def gete_2(q):
    e_1 = q[1]
    e_2 = (3 - GAMMA)*q[1]*q[1]/(2*q[0]) + (GAMMA - 1)*q[2]
    e_3 = q[1]*q[2]/q[0]*GAMMA - q[1]**3/(2*q[0]**2)*(GAMMA - 1)
    return np.vstack([e_1, e_2, e_3])

def ftcs2(delt, delx, q_in, inlet_qt, outlet_qt):
    n = len(q_in[0])
    q_next = q_in.copy()
    q_next[:, 0] = getq(inlet_qt)[:, 0]
    q_next[:, -1] = getq(outlet_qt)[:, 0]
    q_next[:, 1:2] = q_in[:, 1:2] - delt*diff_1(gete_2(q_in), 1, 2, delx) + 0.01*delx**2*diff_2(q_in, 1, 2, delx) + 0.001*delx**4*diff_4_for(q_in, 1, 2, delx)
    q_next[:, 2:-2] = q_in[:, 2:-2] - delt*diff_1(gete_2(q_in), 2, n - 2, delx) + 0.01*delx**2*diff_2(q_in, 2, n - 2, delx) + 0.001*delx**4*diff_4_cen(q_in, 2, n - 2, delx)
    q_next[:, -2:-1] = q_in[:, -2:-1] - delt*diff_1(gete_2(q_in), n - 2, n - 1, delx) + 0.01*delx**2*diff_2(q_in, n - 2, n - 1, delx) + 0.001*delx**4*diff_4_bac(q_in, n - 2, n - 1, delx)
    return q_next

def lax_fed(delt, delx, q_in, inlet_qt, outlet_qt):
    n = len(q_in[0])
    q_next = q_in.copy()
    q_next[:, 0] = getq(inlet_qt)[:, 0]
    q_next[:, -1] = getq(outlet_qt)[:, 0]
    q_next[:, 1:-1] = 0.5*(q_in[:, 2:] + q_in[:, 0:-2]) - 0.5*delt/delx*(gete_2(q_in[:, 2:]) - gete_2(q_in[:, 0:-2]))
    #print q_next[1]
    return q_next

def getstag(qt_in):
    T = qt_in[2, 0]/(qt_in[0, 0]*R)
    T0 = T + qt_in[1, 0]**2/(2*GAMMA*R)*(GAMMA - 1)
    p0 = qt_in[2, 0]*(T0/T)**(GAMMA/(GAMMA - 1))
    return p0, T0

def inlet_1(delt, delx, qt_in, p0 = 0, T0 = 0):
    if (p0 == 0 or T0 == 0):
        p0, T0 = getstag(qt_in)
    u = qt_in[1, 1]
    T = T0 - u*u/(2*R*GAMMA)*(GAMMA - 1)
    p = p0*(T/T0)**(GAMMA/(GAMMA - 1))
    rho = p/(R*T)
    return np.array([rho, u, p])

def inlet_2(delt, delx, qt_in, p0 = 0, T0 = 0):
    if (p0 == 0 or T0 == 0):
        p0, T0 = getstag(qt_in)
    c0 = (GAMMA*qt_in[2, 0]/qt_in[0, 0])**0.5
    zeta_p = qt_in[1, 0] + 2*c0/(GAMMA - 1)
    c1 = (GAMMA*qt_in[2, 1]/qt_in[0, 1])**0.5
    zeta_m1 = qt_in[1, 1] - 2*c1/(GAMMA - 1)
    zeta_m0 = qt_in[1, 0] - 2*c0/(GAMMA - 1)
    zeta_m = zeta_m0 - (qt_in[1, 0] - c0)*delt/delx*(zeta_m1 - zeta_m0)
    u = (zeta_p + zeta_m)/2
    T = T0 - u*u/(2*R*GAMMA)*(GAMMA - 1)
    p = p0*(T/T0)**(GAMMA/(GAMMA - 1))
    rho = p/(R*T)
    return np.array([rho, u, p])


def outlet_1(delt, delx, qt_in, pa = 0):
    if (pa == 0):
        pa = qt_in[2, -1]
    rho = qt_in[0, -2]*pa/qt_in[2, -2] 
    return np.array([rho, qt_in[1, -2], pa])

def outlet_2(delt, delx, qt_in, pa = 0):
    if (pa == 0):
        pa = qt_in[2, -1]
    zeta0_1 = qt_in[2, -2]/(qt_in[0, -2]**GAMMA)
    zeta0_0 = qt_in[2, -1]/(qt_in[0, -1]**GAMMA)
    zeta0 = zeta0_0 - qt_in[1, -1]*delt/delx*(zeta0_0 - zeta0_1)
    rho_a = (pa/zeta0)**(1/GAMMA)
    c0 = (GAMMA*pa/rho_a)**0.5
    c1 = (GAMMA*qt_in[2, -2]/qt_in[0, -2])**0.5
    zeta_p1 = qt_in[1, -2] + 2*c1/(GAMMA - 1)
    zeta_p0 = qt_in[1, -1] + 2*c0/(GAMMA - 1)
    zeta_p = zeta_p0 - (qt_in[1, -1] + c0)*delt/delx*(zeta_p0 - zeta_p1)
    u = zeta_p - 2*c0/(GAMMA - 1)
    return np.array([rho_a, u, pa])

def in_shock1(delt, delx, qt_in, p0 = 0, T0 = 0): # fixed bc
    return qt_in[:, 0]

def out_shock1(delt, delx, qt_in, p0 = 0, T0 = 0): # fixed bc
    return qt_in[:, -1]

def in_shock2(delt, delx, qt_in, p0 = 0, T0 = 0): #reflection bc
    zeta0 = qt_in[2, 0]/(qt_in[0, 0])**GAMMA
    c0 = (GAMMA*qt_in[2, 0]/qt_in[0, 0])**0.5
    c1 = (GAMMA*qt_in[2, 1]/qt_in[0, 1])**0.5
    zetam_0 = -1*2/(GAMMA - 1)*c0
    zetam_1 = -1*2/(GAMMA - 1)*c1
    zetam = zetam_0 + c0*delt/delx*(zetam_1 - zetam_0)
    c0 = -1*zetam*(GAMMA - 1)/2.0
    rho = (c0**2/(GAMMA*zeta0))**(1.0/(GAMMA - 1))
    p = zeta0 * rho**GAMMA
    return np.array([rho, 0.0, p])

def out_shock2(delt, delx, qt_in, p0 = 0, T0 = 0):
    zeta0 = qt_in[2, -1]/(qt_in[0, -1])**GAMMA
    c0 = (GAMMA*qt_in[2, -1]/qt_in[0, -1])**0.5
    c1 = (GAMMA*qt_in[2, -2]/qt_in[0, -2])**0.5
    zetap_0 = 1*2/(GAMMA - 1)*c0
    zetap_1 = 1*2/(GAMMA - 1)*c1
    zetap = zetap_0 - c0*delt/delx*(zetap_0 - zetap_1)
    c0 = 0.5*zetap*(GAMMA - 1)
    rho = (c0**2/(GAMMA*zeta0))**(1.0/(GAMMA - 1))
    p = zeta0 * rho**GAMMA
    return np.array([rho, 0.0, p])

def pltfig(name, title, xlabel, ylabel, xdata, ydata, legend, log = 0, loc = 1):
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
    plt.legend(loc = loc)
    plt.savefig(name)
    plt.close()

def solve(method, delx, delt, qt_init, tmax, inletbc, outletbc, p0 = 0, T0 = 0, pa = 0):
    ntimes = (int)(tmax/delt)
    qt_matrix = np.zeros((3, len(qt_init[0]), ntimes + 1))
    qt_matrix[:, :, 0] = qt_init
    q_old = getq(qt_init)    
    for i in range(ntimes):
        inlet = inletbc(delt, delx, qt_matrix[:, :, i], p0, T0)
        outlet = outletbc(delt, delx, qt_matrix[:, :, i], pa)
        q_next = method(delt, delx, q_old, inlet, outlet)
        qt_matrix[:, :, i+1] = getq_tilda(q_next)
        q_old = q_next.copy()
        qt_old = getq_tilda(q_old)
    return qt_matrix

def animator(X,Y, name):
    fig = plt.figure()
    ax = plt.axes(xlim=(np.amin(X), np.amax(X)), ylim=(0.9*np.amin(Y), 1.5*np.amax(Y)))
    line, = ax.plot([], [], lw=2)
    
    def init():
        line.set_data([],[])
        return line,
    def animate(i):
        x = X
        y = Y[:,i*100]
        line.set_data(x, y)
        return line,
    
    an = anim.FuncAnimation(fig, animate, init_func=init, frames=len(Y[0, :])/100, interval= 0.1, blit=True)
    an.save(name + '.mp4', fps=100, extra_args=['-vcodec', 'libx264'])

def q1_2_case(method, delt, delx, qt_init, tmax, inletbc, outletbc, p0, T0, pa, case, m_name):
    qt_matrix = solve(method, delx, delt, qt_init, tmax, inletbc, outletbc, p0, T0, pa)
    ntimes = (int)(tmax/delt)
    nt = ntimes/10
    x = np.linspace(0, 1, len(qt_init[0]))
    delt_x = delt/delx
    pltfig(m_name + str(case) + "_1.png", "Graph of density for delt/delx = " + str(delt_x), "x", 'density', [x, x, x, x], qt_matrix[0, :, (nt, 2*nt, 5*nt/2, 3*nt)], [str(nt*delt) + 's', str(2*nt*delt) + 's', str(2.5*nt*delt) + 's', str(3*nt*delt) + 's'])
    pltfig(m_name + str(case) + "_2.png", "Graph of density for delt/delx = " + str(delt_x), "x", 'density$', [x, x, x, x], qt_matrix[0, :, (4*nt, 5*nt, 6*nt, 13*nt/2)], [str(4*nt*delt) + 's', str(5*nt*delt) + 's', str(6*nt*delt) + 's', str(13*nt/2*delt) + 's'])
    pltfig(m_name + str(case) + "_3.png", "Graph of density for delt/delx = " + str(delt_x), "x", 'density$', [x, x, x, x], qt_matrix[0, :, (7*nt, 8*nt, 9*nt, 10*nt)], [str(7*nt*delt) + 's', str(8*nt*delt) + 's', str(9*delt) + 's', str(10*nt*delt) + 's'])
    pltfig(m_name + str(case) + "_4.png", "Graph of velocity for delt/delx = " + str(delt_x), "x", 'u', [x, x, x, x], qt_matrix[1, :, (nt, 2*nt, (int)(2.5*nt), 3*nt)], [str(nt*delt) + 's', str(2*nt*delt) + 's', str((int)(2.5*nt)*delt) + 's', str(3*nt*delt) + 's'], 0, 2)
    pltfig(m_name + str(case) + "_5.png", "Graph of velocity for delt/delx = " + str(delt_x), "x", 'u', [x, x, x], qt_matrix[1, :, (4*nt, 5*nt, 6*nt)], [str(4*nt*delt) + 's', str(5*nt*delt) + 's', str(6*nt*delt) + 's'], 0, 4)
    pltfig(m_name + str(case) + "_6.png", "Graph of velocity for delt/delx = " + str(delt_x), "x", 'u', [x, x, x, x], qt_matrix[1, :, (7*nt, 8*nt, 9*nt, 10*nt)], [str(7*nt*delt) + 's', str(8*nt*delt) + 's', str(9*nt*delt) + 's', str(10*nt*delt) + 's'], 0, 2)
    pltfig(m_name + str(case) + "_7.png", "Graph of pressure for delt/delx = " + str(delt_x), "x", 'p', [x, x, x, x], qt_matrix[2, :, (nt, 2*nt, (int)(2.5*nt), 3*nt)], [str(nt*delt) + 's', str(2*nt*delt) + 's', str((int)(2.5*nt)*delt) + 's', str(3*nt*delt) + 's'])
    pltfig(m_name + str(case) + "_8.png", "Graph of pressure for delt/delx = " + str(delt_x), "x", 'p', [x, x, x], qt_matrix[2, :, (4*nt, 5*nt, 6*nt)], [str(4*nt*delt) + 's', str(5*nt*delt) + 's', str(6*nt*delt) + 's'], 0, 3)
    pltfig(m_name + str(case) + "_9.png", "Graph of pressure for delt/delx = " + str(delt_x), "x", 'p', [x, x, x, x], qt_matrix[2, :, (7*nt, 8*nt, 9*nt, 10*nt)], [str(7*nt*delt) + 's', str(8*nt*delt) + 's', str(9*nt*delt) + 's', str(10*nt*delt) + 's'], 0, 3)
    if case == 1:
        animator(x, qt_matrix[0,:, :], 'density' + m_name + str(case))
        animator(x, qt_matrix[1,:, :], 'velocity' + m_name + str(case))
        animator(x, qt_matrix[2,:, :], 'pressure' + m_name + str(case))

def q1_2(ques):
    p0 = 101325.0
    T0 = 300.0
    pa = 84000.0
    Ta = 300.0
    rho_a = pa/(R*Ta)
    xmax = 1.0
    delx = 0.001
    nx = (int)(xmax/delx) + 1
    qt_init = np.zeros((3, nx))
    qt_init[0] += rho_a
    qt_init[2] += pa
    tmax = 0.01
    if (ques == 1):
        q1_2_case(ftcs2, 0.0001*delx, delx, qt_init, tmax, inlet_2, outlet_2, p0, T0, pa, 1, "FTCS2_")
        q1_2_case(ftcs2, 0.0003*delx, delx, qt_init, tmax, inlet_2, outlet_2, p0, T0, pa, 2, "FTCS2_")
        q1_2_case(ftcs2, 0.0004*delx, delx, qt_init, tmax, inlet_2, outlet_2, p0, T0, pa, 3, "FTCS2_")
    elif (ques == 2):
        q1_2_case(lax_fed, 0.0001*delx, delx, qt_init, tmax, inlet_2, outlet_2, p0, T0, pa, 1, "lax_fed_")
        q1_2_case(lax_fed, 0.0005*delx, delx, qt_init, tmax, inlet_2, outlet_2, p0, T0, pa, 2, "lax_fed_")
        q1_2_case(lax_fed, 0.001*delx, delx, qt_init, tmax, inlet_2, outlet_2, p0, T0, pa, 3, "lax_fed_")
        q1_2_case(lax_fed, 0.002*delx, delx, qt_init, tmax, inlet_2, outlet_2, p0, T0, pa, 4, "lax_fed_")

def q1():
    q1_2(1)

def q2():
    q1_2(2)

def q3():
    p_l = 1.0
    rho_l = 1.0
    p_r = 0.1
    rho_r = 0.125
    tmax = 0.2
    xmax = 1.0
    delx = 0.001
    delt = 0.01*delx
    nx = (int)(xmax/delx)
    qt_init = np.zeros((3, nx))
    qt_init[0, 0:(nx/2)] = rho_l
    qt_init[0, (nx/2):] = rho_r
    qt_init[2, 0:(nx/2)] = p_l
    qt_init[2, (nx/2):] = p_r
    x = np.linspace(0, 1, len(qt_init[0]))
    qt_matrix = solve(ftcs2, delx, delt, qt_init, tmax, in_shock2, out_shock2)
    pltfig("Sh_tube_den.png", "Graph of density at t = 0.2", 'x', 'rho', [x], [qt_matrix[0, :, -1]], ['density'])
    pltfig("Sh_tube_vel.png", "Graph of velocity at t = 0.2", 'x', 'u', [x], [qt_matrix[1, :, -1]], ['velocity'])
    pltfig("Sh_tube_p.png", "Graph of pressure at t = 0.2", 'x', 'p', [x], [qt_matrix[2, :, -1]], ['pressure'])
    pltfig("Sh_tube_en.png", "Graph of internal energy at t = 0.2", 'x', 'e', [x], [qt_matrix[2, :, -1]/(qt_matrix[0, :, -1]*(GAMMA - 1))], ['energy'])
    animator(x, qt_matrix[0,:, :], 'density_sh')
    animator(x, qt_matrix[1,:, :], 'velocity_sh')
    animator(x, qt_matrix[2,:, :], 'pressure_sh')
    animator(x, qt_matrix[2, :, :]/(qt_matrix[0, :, :]*(GAMMA - 1)), 'pressure_sh')


def main():
    q1()
    q2()
    q3()

if __name__=='__main__':
    main()