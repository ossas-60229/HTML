import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rand


def generate_p10(size: int = 32) -> (np.ndarray, np.ndarray):
    X = np.zeros(size)
    Y = np.zeros(size)
    for i in range(size):
        X[i] = rand.uniform(-1, 1)
        # by uniform distribution
    X.sort()
    for i in range(size):
        Y[i] = np.sign(X[i])
        if rand.uniform(0, 10) > 9:
            Y[i] = -Y[i]
        # noise
    return X, Y
    pass


def generate_p11():
    return generate_p10(8)
    pass


def generate_p12():
    return generate_p11()
    pass


def Ein_calculate(
    X: np.ndarray, Y: np.ndarray, s: int, theta_x: int, DP: dict
) -> float:
    if (s, theta_x) in DP:
        return DP[(s, theta_x)]
    elif theta_x > 0:
        tmp = Ein_calculate(X, Y, s, theta_x - 1, DP)
        theta_f = -1
        if theta_x > 1:
            theta_f = (X[theta_x - 1] + X[theta_x - 2]) / 2
        theta_now = (X[theta_x - 1] + X[theta_x]) / 2
        i = theta_x - 1
        while (i < X.shape[0]) and (X[i] <= theta_now):
            tmp += abs(Y[i] - s * np.sign(X[i] - theta_now)) - abs(
                Y[i] - s * np.sign(X[i] - theta_f)
            )
            i += 1
            # same x
        DP[(s, theta_x)] = tmp
        return tmp
    elif theta_x == 0:
        theta = -1
        tmp = 0
        size = X.shape[0]
        for i in range(size):
            tmp += abs(Y[i] - s * np.sign(X[i] - theta))
        DP[(s, theta_x)] = tmp
        return tmp
    pass


def Ein_calculate_brut(
    X: np.ndarray, Y: np.ndarray, s: int, theta: int, 
) -> float:
    tmp = 0
    size = X.shape[0]
    for i in range(size):
        tmp += abs(Y[i] - s * np.sign(X[i] - theta))
    return tmp

def Eout_cal(s,theta):
    return 0.5 - 0.4*s + 0.4*s*abs(theta)
def exec_10():
    X, Y = generate_p10()
    DP = dict()
    for i in range(2):
        for j in range(32):
            s = -1
            if i == 0:
                s = 1
            DP[(s, j)] = Ein_calculate(X, Y, s, j, DP)
    fins, finj = 1,0
    fuck = DP[(fins,finj)]
    for i in range(2):
        for j in range(32):
            s = -1
            if i == 0:
                s = 1
            if fuck > DP[(s, j)]:
                fuck,fins,finj = DP[(s, j)],s,j
    #print(fuck,fins,finj)
    fin_theta = -1
    if finj > 1:
        fin_theta = (X[finj - 1] + X[finj]) / 2
    Einmin = fuck / X.shape[0]
    eout = Eout_cal(fins,fin_theta)
    DP.clear()
    return Einmin, eout
    pass

def exec_11():
    X, Y = generate_p11()
    DP = dict()
    for i in range(2):
        for j in range(X.shape[0]):
            s = -1
            if i == 0:
                s = 1
            DP[(s, j)] = Ein_calculate(X, Y, s, j, DP)
    fins, finj = 1,0
    fuck = DP[(fins,finj)]
    for i in range(2):
        for j in range(X.shape[0]):
            s = -1
            if i == 0:
                s = 1
            #print((s,j))
            #print((DP[(s, j)], Ein_calculate_brut(X, Y, s, j)))
            #brut = Ein_calculate_brut(X, Y, s, j)
            #if brut != DP[(s, j)]:
                #print("fuck you")
            if fuck > DP[(s, j)]:
                fuck,fins,finj = DP[(s, j)],s,j
    #print(fuck,fins,finj)
    fin_theta = -1
    if finj > 1:
        fin_theta = (X[finj - 1] + X[finj]) / 2
    Einmin = fuck / X.shape[0]
    eout = Eout_cal(fins,fin_theta)
    DP.clear()
    return Einmin, eout
def exec_12():
    X, Y = generate_p12()
    s,theta = np.sign(rand.uniform(-1,1)), rand.uniform(-1,1)
    Ein = Ein_calculate_brut(X, Y, s, theta)
    eout = Eout_cal(s,theta)
    return Ein/X.shape[0], eout
def main():
    repeat = 2000
    Eins = []
    eouts = []
    medians = []
    for i in range(repeat):
        #Ein, eout = exec_10()
        #Ein, eout = exec_11()
        Ein, eout = exec_12()
        Eins.append(Ein)
        eouts.append(eout)
        medians.append(eout - Ein) 
    # scatter
    median = np.median(medians)
    print("The median is ", median)
    plt.scatter(Eins, eouts)
    plt.show()

if __name__ == "__main__":
    main()
