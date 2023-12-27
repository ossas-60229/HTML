import numpy as np
import math
import matplotlib.pyplot as plt
def gen_9(seed:int=0,size:int=4352)->(np.ndarray,np.ndarray):
    np.random.seed(seed)
    Y = np.random.randint(0,2,size)
    num_n, num_p = 0,0
    for i in range(size):
        if Y[i] == 0:
            Y[i] = -1
            num_n += 1
        else:
            Y[i] = 1
            num_p += 1
    # flip the coin
    np.random.seed(seed)
    mean = [3,2]
    cov = [[0.4,0],[0,0.4]]
    x_p = np.random.multivariate_normal(mean,cov,num_p)
    np.random.seed(seed)
    mean = [5,0]
    cov = [[0.6,0],[0,0.6]]
    x_n = np.random.multivariate_normal(mean,cov,num_n)
    X = []
    for i in range(size):
        if Y[i] == 1:
            X.append([1,x_p[num_p-1][0],x_p[num_p-1][1]])
            num_p -= 1
        else:
            X.append([1,x_n[num_n-1][0],x_n[num_n-1][1]])
            num_n -= 1
    return np.array(X),np.array(Y)
# generate the testcase for problem 9
def exec_9(seed:int=0,size:int=4352)->float:
    # linear regression
    X,Y = gen_9(seed=seed,size=size)
    weight = np.zeros(3)
    spllit = 256
    train_X = X[:spllit]
    train_Y = Y[:spllit]
    test_X = X[spllit:]
    test_Y = Y[spllit:]
    XTX = np.dot(np.transpose(train_X),train_X)
    XTXI = np.linalg.pinv(XTX)
    # psuedo inverse
    weight = np.dot(XTXI,np.dot(np.transpose(train_X),train_Y))
    error = 0
    for i in range(len(test_Y)):
        # squared error
        error += (test_Y[i] - np.dot(weight,test_X[i]))**2
    average_error = error/len(test_Y)
    return average_error
# problem 9
def exec_10(seed:int=0,size:int=4352)->float:
    # linear regression
    X,Y = gen_9(seed=seed,size=size)
    weight = np.zeros(3)
    spllit = 256
    train_X = X[:spllit]
    train_Y = Y[:spllit]
    test_X = X[spllit:]
    test_Y = Y[spllit:]
    XTX = np.dot(np.transpose(train_X),train_X)
    XTXI = np.linalg.pinv(XTX)
    # psuedo inverse
    weight = np.dot(XTXI,np.dot(np.transpose(train_X),train_Y))
    error = 0
    for i in range(len(test_Y)):
        # 0/1 error
        pred = np.dot(weight,test_X[i])
        if pred > 0:
            pred = 1
        else:
            pred = -1
        if pred != test_Y[i]:
            error += 1
    average_error = error/len(test_Y)
    return average_error
# problem 10
def exec_10(seed:int=0,size:int=4352)->float:
    # linear regression
    X,Y = gen_9(seed=seed,size=size)
    weight = np.zeros(3)
    spllit = 256
    train_X = X[:spllit]
    train_Y = Y[:spllit]
    test_X = X[spllit:]
    test_Y = Y[spllit:]
    XTX = np.dot(np.transpose(train_X),train_X)
    XTXI = np.linalg.pinv(XTX)
    # psuedo inverse
    weight = np.dot(XTXI,np.dot(np.transpose(train_X),train_Y))
    error = 0
    for i in range(len(test_Y)):
        # 0/1 error
        pred = np.dot(weight,test_X[i])
        if pred > 0:
            pred = 1
        else:
            pred = -1
        if pred != test_Y[i]:
            error += 1
    average_error = error/len(test_Y)
    return average_error
# problem 10
def exec_11(seed:int=0,size:int=4352)->(float,float):
    # linear regression
    X,Y = gen_9(seed=seed,size=size)
    weight = np.zeros(3)
    spllit = 256
    train_X = X[:spllit]
    train_Y = Y[:spllit]
    test_X = X[spllit:]
    test_Y = Y[spllit:]
    XTX = np.dot(np.transpose(train_X),train_X)
    XTXI = np.linalg.pinv(XTX)
    # psuedo inverse
    weight = np.dot(XTXI,np.dot(np.transpose(train_X),train_Y))
    error = 0
    for i in range(len(test_Y)):
        # 0/1 error
        pred = np.dot(weight,test_X[i])
        if pred > 0:
            pred = 1
        else:
            pred = -1
        if pred != test_Y[i]:
            error += 1
    average_error_A = error/len(test_Y)
    # linear regression
    lr = 0.1
    T = 500
    weight = np.zeros(3)
    for t in range(T):
        gradient = np.zeros(3)
        for i in range(len(train_X)):
            pred = np.dot(weight,train_X[i])
            error = np.exp(-1*train_Y[i]*pred)/(1+np.exp(-1*train_Y[i]*pred))
            # cross entropy
            if pred != train_Y[i]:
                gradient += -1*error*train_X[i]*train_Y[i]
        gradient /= len(train_X)
        gradient /= np.linalg.norm(gradient)+ 1
        # prevent 0
        weight -= lr*gradient
    err = 0
    for i in range(len(test_Y)):
        # 0/1 error
        pred = np.dot(weight,test_X[i])
        if pred > 0:
            pred = 1
        else:
            pred = -1
        if pred != test_Y[i]:
            err += 1
    average_error_B = err/len(test_Y)
    return average_error_A,average_error_B
# problem 10
def exec_12(seed:int=0,size:int=4352)->(float,float):
    # linear regression
    X,Y = gen_9(seed=seed,size=size)
    # new data
    weight = np.zeros(3)
    spllit = 256
    train_X = X[:spllit]
    train_Y = Y[:spllit]
    np.random.seed(seed)
    mean = [0,6]
    cov = [[0.1,0],[0,0.3]]
    news = 16
    tmp = np.random.multivariate_normal(mean,cov,news)
    Y2 = []
    X2 = []
    for i in range(news):
        X2.append([1,tmp[i][0],tmp[i][1]])
        Y2.append(1)
    X2 = np.array(X2)
    Y2 = np.array(Y2)
    train_X = np.concatenate((train_X,X2),axis=0)
    train_Y = np.concatenate((train_Y,Y2),axis=0)
    test_X = X[spllit:]
    test_Y = Y[spllit:]
    XTX = np.dot(np.transpose(train_X),train_X)
    XTXI = np.linalg.pinv(XTX)
    # original case
    # psuedo inverse
    weight = np.dot(XTXI,np.dot(np.transpose(train_X),train_Y))
    error = 0
    for i in range(len(test_Y)):
        # 0/1 error
        pred = np.dot(weight,test_X[i])
        if pred > 0:
            pred = 1
        else:
            pred = -1
        if pred != test_Y[i]:
            error += 1
    average_error_A = error/len(test_Y)
    # linear regression
    lr = 0.1
    T = 500
    weight = np.zeros(3)
    for t in range(T):
        gradient = np.zeros(3)
        for i in range(len(train_X)):
            pred = np.dot(weight,train_X[i])
            error = np.exp(-1*train_Y[i]*pred)/(1+np.exp(-1*train_Y[i]*pred))
            # cross entropy
            if pred != train_Y[i]:
                gradient += -1*error*train_X[i]*train_Y[i]
        gradient /= len(train_X)
        gradient /= np.linalg.norm(gradient)+ 1
        # prevent 0
        weight -= lr*gradient
    err = 0
    for i in range(len(test_Y)):
        # 0/1 error
        pred = np.dot(weight,test_X[i])
        if pred > 0:
            pred = 1
        else:
            pred = -1
        if pred != test_Y[i]:
            err += 1
    average_error_B = err/len(test_Y)
    return average_error_A,average_error_B
# problem 10
def main():
    times = 128
    err_l = []
    err_la = []
    err_lb = []
    #print("problem 9:")
    #print("problem 10:")
    for i in range(times):
        #err = exec_9(seed=i,size=4352)
        #err = exec_10(seed=i,size=4352)
        #err_l.append(err)
        err_a,err_b = exec_12(seed=i,size=4352)
        err_la.append(err_a)
        err_lb.append(err_b)
    err_l.sort()
    err_la.sort()
    err_lb.sort()
    #print("the median is ",err_l[times//2])
    print("the median of A is ",err_la[times//2])
    print("the median of B is ",err_lb[times//2])
    #plt.hist(err_l)
    plt.scatter(err_la,err_lb)
    plt.show()
if __name__ == "__main__":
    main()
