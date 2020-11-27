import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=16)

MAX_ITE = 1000

#constants defined by the problem
x_c = 1.0
y_c = 2.0
#after calculating the derivatives, only sigma^2 is obtained
sigma_x2 = np.power(0.75, 2) #must be != 0
sigma_y2 = np.power(0.5, 2) #must be != 0
sol = np.matrix([1, 2]).T #solution for computing letter d)

#auxiliary variable and functions to avoid recomputations
frac_x = 0
frac_y = 0
f_exp = 0

#choose which constants to use. var=0 : x, var=1: y
def func_frac(xy, var=0):
    if var == 0:
        return (xy - x_c)/sigma_x2
    elif var == 1:
        return (xy - y_c)/sigma_y2
    return np.NaN

#part of the function that has the exponential,
#which remains unchanged after calculating derivatives
def func_exp(x, y):
    fracx = - frac_x * (x - x_c)/2
    fracy = - frac_y * (y - y_c)/2
    return np.exp(fracx + fracy)

#partial derivative with respect to x
#NOTE: partial with respect to x and y have the same structure.
#Hence, to compute with respect to x:
#   `partial_f(x, y, 0)`
#with respect to y:
#   `partial_f(y, x, 1)`
def partial_f(xy, var=0):
    ret = np.NaN

    if var == 0:
        ret = frac_x * f_exp + 2/25 * (xy - x_c)
    elif var == 1:
        ret = frac_y * f_exp + 2/25 * (xy - y_c)

    return ret

#analogous for the second partial derivative
def sec_partial_f(var=0):
    ret = np.NaN

    if var == 0:
        ret = f_exp * (1/sigma_x2 - np.power(frac_x, 2)) + 2/25
    if var == 1:
        ret = f_exp * (1/sigma_y2 - np.power(frac_y, 2)) + 2/25

    return ret

def xy_partial_f():
    return frac_x * frac_y * f_exp

#functions for plotting 3D graph
def func(x, y):
    fracx = - np.power(x - x_c, 2)/(2*sigma_x2)
    fracy = - np.power(y - y_c, 2)/(2*sigma_y2)

    return 1 - np.exp(fracx + fracy) + 1/25 * ( np.power(x - x_c, 2) + np.power(y - y_c, 2) )

def plot(vn, name):
    gtruth = np.matrix([1, 2, func(1, 2)]).T
    plt.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x = np.arange(-2, 4, 0.1)
    y = np.arange(0, 4, 0.1)
    x, y = np.meshgrid(x, y)
    z = func(x, y)

    surf = ax.plot_surface(x, y, z, cmap = plt.cm.winter,
            linewidth=0, antialiased=False, alpha=0.15)

    ax.set_zlim(0, 1.5)
    ax.set_xlabel("x", fontsize="12", fontweight='bold')
    ax.set_xticks([-2, 0, 2, 4])
    ax.set_yticks([0, 2, 4])
    ax.set_ylabel("y", fontsize="12", fontweight='bold')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.plot([vn[0], gtruth[0, 0]], [vn[1], gtruth[1, 0]],
            [func(vn[0], vn[1]), gtruth[2, 0]], color='r')

    plt.savefig(name, bbox_inches='tight')

#################### NEWTON METHOD ####################
v0 = np.matrix([0.0, 0.0]).T #initial guess
eps = 10**-8 #tolerance

def newton_method(v0, eps, make_plot=False):
    vn = np.copy(v0) #v_next
    H = np.matrix([[0.0, 0.0], [0.0, 0.0]])
    global frac_x
    global frac_y
    global f_exp

    ite_count = 0
    norm_dif = 2*eps

    print("===== Iteration #" + str(ite_count) + " =====")
    print("v_" + str(ite_count) + ":")
    print(vn)
    print("Error (Euclidian Norm): " + str(np.linalg.norm(sol - vn)))
    if (make_plot):
        plot(v0, "q1-ite" + str(ite_count))

    while (ite_count < MAX_ITE and norm_dif > eps):
        #updates iteration and pre-computes stuff
        v0 = np.copy(vn)
        frac_x = func_frac(v0[0])
        frac_y = func_frac(v0[1], 1)
        f_exp = func_exp(v0[0], v0[1])

        #updates vn
        vn[0] = partial_f(v0[0])
        vn[1] = partial_f(v0[1], 1)

        #constructs Hessian Matrix
        H[0,0] = sec_partial_f()
        H[0,1] = H[1,0] = xy_partial_f()
        H[1,1] = sec_partial_f(1)

        #computes newton iteration
        H = np.linalg.inv(H)
        vn = v0 - np.matmul(H, vn)
        norm_dif = np.linalg.norm(vn - v0) #uses euclidian norm to have distance intuition

        #print info
        ite_count += 1
        print("\n===== Iteration #" + str(ite_count) + " =====")
        print("v_" + str(ite_count) + ":")
        print(vn)
        print("Error (Euclidian Norm): " + str(np.linalg.norm(sol - vn)))
        print("Step Distance (Euclidian Norm): " + str(norm_dif))
        if (make_plot):
            plot(vn, "q1-ite" + str(ite_count))

    return vn


ret = newton_method(v0, eps, True)

#plot( np.matrix([0, 0]).T )
