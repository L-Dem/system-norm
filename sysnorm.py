# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 16:17:49 2018

@author: Waterdrop
"""

import numpy as np
from numpy import linalg as lin
import matplotlib.pyplot as plt


class GridNet:
    def __init__(self, n, m):
        self.nrow = n
        self.ncol = m
    
    def toindex(self, i, j):
        return i*self.ncol + j
    
    def toij(self, index):
        j = index % self.ncol
        i = int(index / self.ncol)
        result = (i, j)
        return result

    def ij_valid(self, i, j):
        result = (0 <= i < self.nrow and 0 <= j < self.ncol)
        return result
    
    def __push_valid(self, ret_list, i, j):
        if self.ij_valid(i, j):
            ret_list.append(self.toindex(i, j))
    
    def neb(self, index):
        i, j = self.toij(index)
        neb = []
        self.__push_valid(neb, i+1, j)
        self.__push_valid(neb, i-1, j)
        self.__push_valid(neb, i, j-1)
        self.__push_valid(neb, i, j+1)
        return neb
    
    def input_vec(self, i, j, dtype=float):
        b = np.zeros(self.N, dtype)
        b[self.toindex(i, j)] = 1
        return b
    
    @property
    def N(self):
        return self.nrow * self.ncol
    
    def laplacian(self, dtype=float):
        n = self.N
        A = np.zeros([n, n], dtype)
        for i in range(n):
            A[i, :][self.neb(i)] = 1
        D = np.diag(np.sum(A, 1))
        return D - A


class SysU2E:
    def G1(L, B, w):
        n, m = L.shape
        if n == m:
            b1 = np.ones(n)
            sI = np.eye(n)*complex(0, w)
            L1 = L + sI + np.diag(B)
            return lin.solve(L1, B) - b1
        return None
    
    def G2(L, B, w, acc=1.0, kappa=1.0):
        n, m = L.shape
        if n == m:
            b1 = np.ones(n)
            B1 = np.diag(B)
            B2 = np.eye(n)*kappa
            sI = np.eye(n)*complex(0, w)
            L1 = acc*L + sI + B1
            L2 = L + sI + B2
            g1 = lin.solve(L1, B)
            g2 = lin.solve(L2, B2)
            return g2.dot(g1) - b1
        return None
    
    def grad1(L, B, w):
        n, m = L.shape
        if n == m:
            b1 = np.matrix(np.ones([n, 1]))
            b2 = np.matrix(B).transpose()
            L1 = np.matrix(L + np.diag(np.ones(n)*complex(0, w) + B))
            x = np.matrix(lin.solve(L1, b2)) - b1
            y = np.matrix(lin.solve(L1*L1, b2))
            gd = 2*np.imag(x.H*y)
            return np.trace(gd)
        return None

    def test_norm(L, B, wrange, acc=1.0, kappa=1.0):
        n1 = [lin.norm(SysU2E.G1(L, B, w)) for w in wrange]
        n2 = [lin.norm(SysU2E.G2(L, B, w, acc, kappa)) for w in wrange]
        return np.array([n1, n2])


class SysV2E:
    def G1(L, B, w):
        n, m = L.shape
        if n == m:
            b1 = np.ones(n)
            L1 = L + np.diag(b1*complex(0, w) + B)
            return lin.solve(L1, b1)
        return None

    def G2(L, B, w, acc=1.0, kappa=1.0):
        n, m = L.shape
        if n == m:
            b1 = np.ones(n); sI = b1*complex(0, w)
            L1 = acc*L + np.diag(sI + B)
            L2 = L + np.diag(sI + b1*kappa)
            g1 = lin.solve(L1, b1)
            g2 = lin.solve(L2, np.diag(B))
            return (g2 + np.eye(n)).dot(g1)
        return None
    
    def test_norm(L, B, wrange, acc = 1.0, kappa = 1.0):
        n1 = [lin.norm(SysV2E.G1(L, B, w)) for w in wrange]
        n2 = [lin.norm(SysV2E.G2(L, B, w, acc, kappa)) for w in wrange]
        return np.array([n1, n2])


def numeric_diff(val, dx):
    n = len(val)
    return (val[1:]-val[0:n-1])/dx


Graph0 = GridNet(8, 8)
L0 = Graph0.laplacian()
b0 = Graph0.input_vec(0, 0)
wmax = 10
dw = 0.001  # 0.001
ws = np.linspace(0, wmax, int(wmax/dw)+1)


def main_v2e():
    ret = SysV2E.test_norm(L0, b0, ws, acc=1, kappa=1.0)
    plt.figure()
    plt.plot(ws, ret[0, :])
    plt.plot(ws, ret[1, :])
    print(ret[0, 0])
    print(ret[1, 0])
    plt.legend(['SingleLayer', 'DoubleLayer'])
    plt.show()


def main_u2e():
    ret = SysU2E.test_norm(L0, b0, ws, acc=10.0, kappa=1.0)
    plt.figure()
    plt.plot(ws, ret[0, :])
    plt.plot(ws, ret[1, :])
    plt.legend(['SingleLayer', 'DoubleLayer'])
    plt.show()


def main_u2e_grad_test():
    ns = np.array([lin.norm(SysU2E.G1(L0, b0, w)) for w in ws])
    ns2 = ns*ns
    gd2 = numeric_diff(ns2, dw) # num diff is one item less
    gd2_algo = np.array([SysU2E.grad1(L0, b0, w) for w in ws])
    gd2_algo = gd2_algo[0:len(gd2)]
    ws2 = ws[0:len(gd2)]
    diff_gd2 = gd2_algo - gd2
    rdif_gd2 = diff_gd2 / gd2
 
    plt.figure()
    plt.plot(ws2, gd2_algo)
    plt.plot(ws2, gd2)
    plt.legend(['grad by algo', 'grad by num diff'])
    
    plt.figure()
    plt.plot(ws2, rdif_gd2)
    plt.title('relative gradient error')
    plt.show()
    return


if __name__ == '__main__':
    main_v2e()
    # main_u2e_grad_test()
    # main_v2e()
