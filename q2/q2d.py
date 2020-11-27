import numpy as np
import time
import matplotlib.pyplot as plt

np.set_printoptions(precision=16)

#constants defined by the problem
MAX_ITE = 1000
p = 15
l = 1
kappa = 100
eps = 10**-8 #tolerance

def f1(v):
    return 7*p*l/2 * np.cos(v[0]) + kappa*(v[1] - 2*v[0])
def f2(v):
    return 5*p*l/2 * np.cos(v[1]) + kappa*(v[2] - 2*v[1] + v[0])
def f3(v):
    return 3*p*l/2 * np.cos(v[2]) + kappa*(v[3] - 2*v[2] + v[1])
def f4(v):
    return p*l/2 * np.cos(v[3]) + kappa*(v[2] - v[3])

def eval_funcs(v, out):
    out[0] = f1(v)
    out[1] = f2(v)
    out[2] = f3(v)
    out[3] = f4(v)

def create_jacob_no_main_diag():
    jacob = np.zeros([4,4])
    for i in range(1, 4):
        jacob[i-1, i] = kappa
        jacob[i, i-1] = kappa
    return jacob

def get_jacobian(jacob, t_vec):
    for i in range(3):
        jacob[i, i] = -(7 - 2*i)*p*l/2 * np.sin(t_vec[i]) - 2*kappa
    jacob[3, 3] = -p*l/2 * np.sin(t_vec[3]) - kappa

def plot_graph(angles, ite, prev=None):
    #calculating points
    x = [0] * 5
    y = [4] * 5
    #leg = ['Wall', 'Mechanical System']

    for i in range(1, 5):
        x[i] = x[i-1] + np.cos(angles[i-1])
        y[i] = y[i-1] - np.sin(angles[i-1])

    plt.plot([0, 0], [0, 5], color='k', label='Wall')

    if prev is not None:
        x_prev = [0] * 5
        y_prev = [4] * 5
        for i in range(1, 5):
            x_prev[i] = x_prev[i-1] + np.cos(prev[i-1])
            y_prev[i] = y_prev[i-1] - np.sin(prev[i-1])
        plt.plot(x_prev, y_prev, marker="o", color='r', alpha=0.3, label='Previous Iteration')

    plt.plot(x, y, marker="o", color='c', label='Mechanical System')

    plt.xticks(np.arange(-1.5, 5))
    plt.legend()
    plt.savefig('q2d-ite' + str(ite), bbox_inches='tight')
    plt.clf()

def broyden(v0, debug=False):
    vn = np.copy(v0)
    prev = np.copy(v0)
    ite = 0

    if (debug):
        plot_graph(vn, ite)

    ite += 1
    #step 1
    jacob = create_jacob_no_main_diag()
    get_jacobian(jacob, vn)
    v = np.copy(vn)
    eval_funcs(vn, v)
    
    #step 2
    A = np.linalg.inv(jacob)

    #step 3
    s = np.matmul(-A, v)
    vn += s
    inf_norm = np.linalg.norm(s, np.inf)

    if (debug):
        print("\n===== Iteration #" + str(ite) + " =====")
        #print("v_" + str(ite) + ":")
        #print(vn)
        #print("\nv_" + str(ite) + "/pi:")
        #print(vn / np.pi)
        latex = str(vn[0,0]/np.pi)
        for i in range(1, len(vn)):
            latex += " \\\\ " + str(vn[i,0]/np.pi)
        print(latex)
        print("||v_" + str(ite) + " - v_" + str(ite-1) + "||_2: " + str(inf_norm))
        #plot_graph(vn, ite, prev)
        plot_graph(vn, ite, vn - s)

    #step 4
    while(ite < MAX_ITE and inf_norm >= eps):
        #prev = np.copy(vn)
        #step 5
        w = np.copy(v)
        eval_funcs(vn, v)
        y = v - w

        #step 6
        z = np.matmul(-A, y)
        #step7
        p = np.matmul(-s.T, z)
        #step 8
        ut = np.matmul(s.T, A)
        #step 9
        A += 1/p * np.matmul(s+z, ut)
        #step 10
        s = np.matmul(-A, v)
        #step 11
        vn += s
        #step 12
        inf_norm = np.linalg.norm(s, np.inf)
        #step 13
        ite += 1

        if (debug):
            print("\n===== Iteration #" + str(ite) + " =====")
            #print("v_" + str(ite) + ":")
            #print(vn)
            #print("\nv_" + str(ite) + "/pi:")
            #print(vn / np.pi)
            latex = str(vn[0,0]/np.pi)
            for i in range(1, len(vn)):
                latex += " \\\\ " + str(vn[i,0]/np.pi)
            print(latex)
            print("||v_" + str(ite) + " - v_" + str(ite-1) + "||_2: " + str(inf_norm))
            #plot_graph(vn, ite, prev)
            plot_graph(vn, ite, vn - s)

    return vn, ite

#initial guess
v0 = np.matrix([0.0, 0.0, 0.0, 0.0]).T

#debug = True will generate graphs
vn, ite = broyden(v0, True)

#benchmarking
n = 100
exec_times = [0]*n
print("\n\n===== BENCHMARKING =====")
for i in range(n):
    print("Execution " + str(i+1) + "/" + str(n))
    start = time.time()
    broyden(v0)
    end = time.time()
    exec_times[i] = end - start

print("\n\n===== RESULTS =====")
print("Solution:")
print(vn)
print("\nSolution / pi:")
print(vn/np.pi)
print("\nNumber of iterations: " + str(ite))
print("Average execution time: " + str(np.mean(exec_times)) + " seconds")
print("Standard deviation: " + str(np.std(exec_times)))
