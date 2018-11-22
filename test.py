import numpy as np
from numpy import linalg as lin
import matplotlib.pyplot as plt
import math

L = np.matrix(([2, -1, -1],
             [-1, 1, 0],
             [-1, 0, 1]))
B = np.diag(np.array([1, 0, 0]))
# gamma_all = [0.0000000000001]
gamma_all = np.arange(0.01, 10, 0.01)
# w_all = np.arange(0, 10, 0.1)
w_all = [0]
y_all = []
y_put_all = []
onet = np.array(([1, 1, 1]))
one = onet.reshape(-1, 1)
norm_single = []
norm_double = []
a = 1
for gamma in gamma_all:
    w = 0
    l_gamma = L + w * np.complex(0, 1) * np.eye(3) + gamma * np.eye(3)
    l_b_single = L + w * np.complex(0, 1) * np.eye(3) + B
    lb_single_inv = lin.inv(l_b_single)
    l_b = a * L + w * np.complex(0, 1) * np.eye(3) + B
    lginv = lin.inv(l_gamma)
    lb_double_inv = lin.inv(l_b)
    K = gamma * lb_double_inv + np.eye(3)
    result_d = np.matmul(np.matmul(np.matmul(onet, K.H), np.matmul(lginv.H, lginv)),
                         np.matmul(np.matmul(lginv, K) + gamma * np.matmul(lb_double_inv, lb_double_inv), one)).imag
    G_s_double = - np.matmul(np.matmul(lginv, gamma * lb_double_inv + np.eye(3)), one)
    result = abs(np.matmul(G_s_double.H, G_s_double))[0, 0]
    # print("double : " + str(result))
    g_s_double_d = gamma * complex(0, 1) * np.matmul(np.matmul(np.matmul(lginv, lginv), lb_double_inv), one)\
                   + gamma * complex(0, 1) * np.matmul(np.matmul(np.matmul(lginv, lb_double_inv), lb_double_inv), one)\
                   + complex(0, 1) * np.matmul(np.matmul(lginv, lginv), one)

    old_gs_d = np.matmul(G_s_double.H, g_s_double_d)
    # print(result_d[0, 0].real)
    # y_all.append(result_d)
    norm_double.append(result)
    y_all.append(old_gs_d.real[0, 0])
    y_put_all.append(result_d[0, 0])
    test = np.matmul(np.matmul(np.eye(3), np.matmul(lginv.H, lginv)),
                     np.matmul(np.matmul(lb_double_inv, lb_double_inv), np.eye(3)))
    G_s_single = np.matmul(lb_single_inv, one)
    result_single = abs(np.matmul(G_s_single.H, G_s_single)[0, 0])
    norm_single.append(result_single)
    # print("single : " + str(result_single))
    acc_first = np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(onet, lb_double_inv), lb_double_inv), L),
                                              np.matmul(lginv, lginv)), lb_double_inv), one)
    g_d_gamma_first = np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(onet, lb_double_inv), lginv), lginv),
                                                    np.eye(3) - gamma * lginv), lb_double_inv), one)  # positive
    g_d_gamma = np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(onet, np.eye(3)), lginv), lginv),
                                    lb_double_inv - gamma * lb_double_inv * lginv - lginv), one)  # negative
    final1 = np.matmul(np.matmul(np.matmul(np.matmul(
        np.matmul(onet, lb_double_inv), lginv), lginv), lb_double_inv - lginv), one)
    final2 = np.matmul(np.matmul(np.matmul(np.matmul(
        np.matmul(np.matmul(onet, lb_double_inv), lginv), lginv), np.eye(3) - 2 * gamma * lginv), lb_double_inv), one)
    final3 = np.matmul(np.matmul(np.matmul(
        np.matmul(onet, lb_double_inv - lginv - gamma * lb_double_inv * lginv), lginv), lginv), one)
    final4 = np.matmul(np.matmul(np.matmul(np.matmul(
        np.matmul(onet, lb_double_inv), lginv), lginv), lb_double_inv - lginv - gamma * lginv * lb_double_inv), one)
    # print(abs(acc_first)[0, 0])
    print(final4[0, 0])
    # print("g(s)norm" + str(result))
    # print("g(s)_d" + str(result_d))
    # print("----")
# plt.subplot(2, 1, 1)
# plt.plot(w_all, norm_single)
# plt.subplot(2, 1, 2)
# plt.plot(w_all, norm_double)
# plt.show()
