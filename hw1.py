import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data_path = "hw1_train.dat"
def read_dat(file_path:str)->np.ndarray:
    fp = open(file_path)
    data = []
    for line in fp.readlines():
        line = line.strip().split()
        if len(line) == 0:
            continue
        line = [1]+[float(i) for i in line]
        #line = [11.26]+[float(i) for i in line]
        # bias
        data.append(line)
    fp.close()
    return np.array(data)
def init_w(size:int)->np.ndarray:
    return np.zeros(size)
def PLA(seed:int, data:np.ndarray):
    data_x = data[:,:-1]
    data_y = data[:,-1]
    #split
    weight = init_w(data_x.shape[1])
    #initialize
    N = data_x.shape[0]
    time = 1
    win, cnt = 0, 0
    while win < 5*N:
        np.random.seed(seed)
        seq = np.random.randint(N, size=10*N*time)[-10*N:]
        for i in seq:
            if data_y[i] * np.dot(weight, data_x[i]) <= 0:
                weight = weight + data_y[i] * data_x[i]
                #update
                cnt += 1
                win = 0
            else:
                win += 1
        time += 1
    return weight, cnt

def PLA_p12(seed:int, data:np.ndarray):
    data_x = data[:,:-1]
    data_y = data[:,-1]
    #split
    weight = init_w(data_x.shape[1])
    #initialize
    N = data_x.shape[0]
    np.random.seed(seed)
    time = 1
    win, cnt = 0, 0
    while win < 5*N:
        np.random.seed(seed)
        seq = np.random.randint(N, size=10*N*time)[-10*N:]
        for i in seq:
            while data_y[i] * np.dot(weight, data_x[i]) <= 0:
                win = 0
                weight = weight + data_y[i] * data_x[i]
                cnt += 1
                #update
            win += 1
        time += 1
    
    return weight, cnt

def data_p9(data:np.ndarray):
    return data
def data_p10(data:np.ndarray):
    ret = data
    ret[:,:-1] = ret[:,:-1] * 11.26
    return ret
def data_p11(data:np.ndarray):
    ret = data
    ret[:, 0] = 11.26
    return ret
def data_p12(data:np.ndarray):
    return data

data = read_dat(data_path)
#data[:,:-1] = data[:,:-1] * 11.26
#data[:,0] = 11.26
data = data_p9(data)
#data = data_p10(data)
#data = data_p11(data)
#data = data_p12(data)
print(data)
updates = []
for i in range(1000):
    w, cnt = PLA(i, data)
    #w, cnt = PLA_p12(i, data)
    updates.append(cnt)
updates.sort()
print("the median is ", updates[len(updates)//2])
print("the mean is ", np.mean(updates))
plt.hist(updates, bins=100)
plt.show()