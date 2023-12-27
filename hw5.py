import numpy as np
import os
import platform
from libsvm.svmutil import *
import matplotlib.pyplot as plt
import random
import math
train_dpath = "./satimage.scale.txt"
test_dpath = "./tsatimage.scale.t"
if platform.system() == "Windows":
    train_dpath = ".\satimage.scale.txt"
    test_dpath = ".\\tsatimage.scale.t"

def exec_p9():
    train_y, train_x = svm_read_problem(train_dpath)
    the_class = 4
    cnt = 0
    for i in range(len(train_y)):
        if train_y[i] == the_class:
            train_y[i] = 1
            cnt = cnt + 1
        else:
            train_y[i] = -1
    test_y, test_x = svm_read_problem(test_dpath)
    for i in range(len(test_y)):
        if test_y[i] == the_class:
            test_y[i] = 1
        else:
            test_y[i] = -1
    args = ["-s", "-t","-d","-c"]
    Cs = ["0.1", "1", "10"]
    Qs = ["2", "3", "4"]
    CQs = []
    for i1 in range(len(Cs)):
        for j1 in range(len(Qs)):
            params = ["0","1",Qs[i1], Cs[j1]]
            arg_str = " "
            for k in range(len(params)):
                arg_str = arg_str + args[k] + " " + params[k] + " "
            arg_str = arg_str + " -g 1 -r 1 -q"
            # add arguments 
            model = svm_train(train_y, train_x, arg_str)
            # start counting the support vectors.
            support_vectors = model.get_SV()
            CQs.append((len(support_vectors), Cs[j1], Qs[i1]))
    # fint the (C,Q) with least support vectors.
    min = CQs[0]
    for i in range(len(CQs)):
        if CQs[i][0] < min[0]:
            min = CQs[i]
    print(min)
    return 0

def exec_p10():
    train_y, train_x = svm_read_problem(train_dpath)
    the_class = 1
    cnt = 0
    for i in range(len(train_y)):
        if train_y[i] == the_class:
            train_y[i] = 1
            cnt = cnt + 1
        else:
            train_y[i] = -1
    test_y, test_x = svm_read_problem(test_dpath)
    for i in range(len(test_y)):
        if test_y[i] == the_class:
            test_y[i] = 1
        else:
            test_y[i] = -1
    args = ["-s", "-t","-c"]
    Cs = ["0.01", "0.1", "1", "10", "100"]
    CEs = []
    for i1 in range(len(Cs)):
        params = ["0","2", Cs[i1]]
        arg_str = " "
        for k in range(len(params)):
            arg_str = arg_str + args[k] + " " + params[k] + " "
        arg_str = arg_str + " -g 1 -q"
        #print(arg_str)
        # add arguments 
        model = svm_train(train_y, train_x, arg_str)
        p_label, p_acc, p_val = svm_predict(test_y, test_x, model)
        #print(p_acc[0])
        CEs.append((p_acc[0], Cs[i1]))
    # fint the C with least error
    max = CEs[0]
    for i in range(len(CEs)):
        if CEs[i][0] > max[0]:
            max = CEs[i]
    print(max)
    return 0

def exec_p11():
    train_y, train_x = svm_read_problem(train_dpath)
    the_class = 1
    cnt = 0
    for i in range(len(train_y)):
        if train_y[i] == the_class:
            train_y[i] = 1
            cnt = cnt + 1
        else:
            train_y[i] = -1
    test_y, test_x = svm_read_problem(test_dpath)
    for i in range(len(test_y)):
        if test_y[i] == the_class:
            test_y[i] = 1
        else:
            test_y[i] = -1
    print(type(train_x))
    sample_num = 200
    times = 1000
    args = ["-s", "-t","-c"]
    Cs = ["0.01", "0.1", "1", "10", "100"]
    chosen = []
    for seed in range(times):
        if seed % 20 == 0:
            print(seed)
        random.seed(seed)
        sample_index = random.sample(range(len(train_y)), sample_num)
        val_x, val_y,t_x, t_y = [], [], [], []
        for i2 in range(len(train_y)):
            if i2 in sample_index:
                val_x.append(train_x[i2])
                val_y.append(train_y[i2])
            else:
                t_x.append(train_x[i2])
                t_y.append(train_y[i2])
        CEs = []
        for i1 in range(len(Cs)):
            params = ["0","2", Cs[i1]]
            arg_str = " "
            for k in range(len(params)):
                arg_str = arg_str + args[k] + " " + params[k] + " "
            arg_str = arg_str + " -g 1 -q"
            #print(arg_str)
            # add arguments 
            #model = svm_train(train_y, train_x, arg_str)
            model = svm_train(t_y, t_x, arg_str)
            p_label, p_acc, p_val = svm_predict(val_y, val_x, model, " -q")
            CEs.append((p_acc[0], Cs[i1]))
        # fint the C with least error
        max = CEs[0]
        for i in range(len(CEs)):
            if CEs[i][0] > max[0]:
                max = CEs[i]
        chosen.append(float(max[1]))
    x = [i - 2 for i in range(len(Cs))]
    y = [0 for i in range(len(Cs))]
    for i in range(len(chosen)):
        for j in range(len(Cs)):
            if chosen[i] == float(Cs[j]):
                y[j] = y[j] + 1
    plt.bar(x,y)
    plt.show()
    print(x)
    print(chosen)
    return 0

def exec_p12():
    train_y, train_x = svm_read_problem(train_dpath)
    the_class = 1
    cnt = 0
    for i in range(len(train_y)):
        if train_y[i] == the_class:
            train_y[i] = 1
            cnt = cnt + 1
        else:
            train_y[i] = -1
    test_y, test_x = svm_read_problem(test_dpath)
    for i in range(len(test_y)):
        if test_y[i] == the_class:
            test_y[i] = 1
        else:
            test_y[i] = -1
    args = ["-s", "-t","-c"]
    Cs = ["0.01", "0.1", "1", "10", "100"]
    CEs = []
    lens = []
    for i1 in range(len(Cs)):
        params = ["0","2", Cs[i1]]
        arg_str = " "
        for k in range(len(params)):
            arg_str = arg_str + args[k] + " " + params[k] + " "
        arg_str = arg_str + " -g 1 -q"
        # add arguments 
        model = svm_train(train_y, train_x, arg_str)
        p_label, p_acc, p_val = svm_predict(test_y, test_x, model)
        sv_coefs = model.get_sv_coef()
        print(len(sv_coefs))
        length = 0
        for k in range(len(sv_coefs)):
            length += sv_coefs[k][0]**2
            #print(sv_coefs[k][0])
        length = math.sqrt(length)
        print(length)
        CEs.append((p_acc[0], Cs[i1]))
        lens.append(length)
    # fint the C with least error
    x = [float(Cs[i]) for i in range(len(Cs))]
    y = lens
    plt.plot(x,y)
    plt.show()
    return 0

def main():
    #exec_p9()
    exec_p10()
    #exec_p11()
    #exec_p12()
if __name__ == "__main__":
    main()