import numpy as np
import time
import matplotlib.pyplot as plt

np.set_printoptions(precision=17)

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
    plt.savefig('q2a-ite' + str(ite), bbox_inches='tight')
    plt.clf()


def newton(v0, debug=False):
    jacob = create_jacob_no_main_diag()
    
    vn = np.copy(v0)
    funcs_v0 = np.copy(v0)
    #using infinity norm to assert that each angle is within the desired tolerance
    inf_norm = eps
    ite = 0
    if (debug):
        plot_graph(v0, ite)

    while(ite < MAX_ITE and inf_norm >= eps):
        get_jacobian(jacob, v0)
        #computes inverse to simulate worst case scenario in order to better
        #compare with the quasi-newton approach
        inv_jacob = np.linalg.inv(jacob) 
        eval_funcs(v0, funcs_v0) #evaluates functions at v0

        vn = v0 - np.matmul(inv_jacob, funcs_v0)
        inf_norm = np.linalg.norm(vn - v0, np.inf)

        #prepares next iteration and updates iteration count
        ite += 1
        if (debug):
            print("\n===== Iteration #" + str(ite) + " =====")
            print("v_" + str(ite) + ":")
            print(vn)
            print("\nv_" + str(ite) + r"/pi:")
            print(vn / np.pi)
            #latex = str(vn[0,0]/np.pi)
            #for i in range(1, len(vn)):
            #    latex += " \\\\ " + str(vn[i,0]/np.pi)
            #print(latex)
            print("\n||v_" + str(ite) + " - v_" + str(ite-1) + "||_2: " + str(inf_norm))
            plot_graph(vn, ite, v0)
        v0 = np.copy(vn)

    return vn, ite

#initial guess
v0 = np.matrix([0.0, 0.0, 0.0, 0.0]).T

#debug = True will generate graphs
vn, ite = newton(v0, True)

#benchmarking
n = 100
exec_times = [0]*n
print("\n\n===== BENCHMARKING =====")
for i in range(n):
    print("Execution " + str(i+1) + "/" + str(n))
    start = time.time()
    newton(v0)
    end = time.time()
    exec_times[i] = end - start

print("\n\n===== RESULTS =====")
print("Solution:")
print(vn)
print("\nSolution / pi:")
print(vn/np.pi)
print("\nFunctions evaluation for solution:")
res = np.matrix([0.0, 0.0, 0.0, 0.0]).T
eval_funcs(vn, res)
print(res)
print("Solution Error (infinity norm): " + str(np.linalg.norm(res, np.inf)))
print("\nNumber of iterations: " + str(ite))
print("Average execution time: " + str(np.mean(exec_times)) + " seconds")
print("Standard deviation: " + str(np.std(exec_times)))
