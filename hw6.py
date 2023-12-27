import numpy as np
import random
from libsvm.svmutil import *
import matplotlib.pyplot as plt


class CARTRegressionTree:
    def __init__(self, max_depth=None, min_group=2):
        self.max_depth = max_depth
        self.min_group = min_group
        self.root = None

    def mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def split(self, X, y, feature_idx, theta):
        left_mask = X[:, feature_idx] <= theta
        right_mask = X[:, feature_idx] > theta
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        return X_left, y_left, X_right, y_right

    def find_best_split(self, X, y):
        best_mse, best_feature_idx, best_theta = float("inf"), None, None
        for feature_idx in range(X.shape[1]):
            uni_vals = np.unique(X[:, feature_idx])
            np.sort(uni_vals)
            thetas = [
                (uni_vals[i] + uni_vals[i + 1]) / 2
                for i in range(len(uni_vals) - 1)
            ]
            for theta in thetas:
                X_left, y_left, X_right, y_right = self.split(X, y, feature_idx, theta)
                num_l, num_r = len(y_left), len(y_right)
                mse_l, mse_r = self.mse(y_left), self.mse(y_right)
                mse = (num_l * mse_l + num_r * mse_r) / (num_l + num_r)
                if mse < best_mse:
                    best_mse = mse
                    best_feature_idx = feature_idx
                    best_theta = theta
        return best_feature_idx, best_theta

    def set_node(self, node, feature_idx, theta, left, right)->dict:
        node["feature_idx"] = feature_idx
        node["theta"] = theta
        node["left"] = left
        node["right"] = right
        return node

    def build_tree(self, X, y, depth):
        if depth == self.max_depth or len(X) < self.min_group:
            return np.mean(y)
        feature_idx, theta = self.find_best_split(X, y)
        if feature_idx is None or theta is None:
            return np.mean(y)
        X_left, y_left, X_right, y_right = self.split(X, y, feature_idx, theta)
        left_child, right_child = self.build_tree(
            X_left, y_left, depth + 1
        ), self.build_tree(X_right, y_right, depth + 1)
        tree = self.set_node(dict(), feature_idx, theta, left_child, right_child)
        return tree

    def fit(self, X, y):
        self.root = self.build_tree(X, y, 0)
        self.max_depth = None
        self.min_group = 1

    def predict_single(self, x, tree):
        if isinstance(tree, (float, int)):
            return tree
        if x[tree["feature_idx"]] <= tree["theta"]:
            return self.predict_single(x, tree["left"])
        else:
            return self.predict_single(x, tree["right"])

    def err_out(self,str):
        print(str)
        exit()
    def predict(self, X):
        if self.root is None:
            self.err_out("Please fit the model first.")
        preds = list()
        for x in X:
            preds.append(self.predict_single(x, self.root))
        return np.array(preds)


def exec_9():
    print("problem 9")
    # load data
    train_y, train_x = svm_read_problem("hw6_train.dat.txt")
    test_y, test_x = svm_read_problem("hw6_test.dat.txt")
    num_leaves, num_train, num_f = 1, len(train_y), len(train_x[0])
    # init
    t_x, t_y = list(), list()
    for i in range(num_train):
        t_y.append(train_y[i])
        tmpx = list()
        for j in range(num_f):
            tmpx.append(train_x[i][j + 1])
        t_x.append(tmpx)
    t_x, t_y = np.array(t_x), np.array(t_y)
    tree = CARTRegressionTree()
    tree.fit(t_x, t_y)
    v_x, v_y = list(), list()
    for i in range(len(test_y)):
        v_y.append(test_y[i])
        tmpx = list()
        for j in range(num_f):
            tmpx.append(test_x[i][j + 1])
        v_x.append(tmpx)
    v_x, v_y = np.array(v_x), np.array(v_y)
    y_pred = tree.predict(v_x)
    # calculate squared error
    total = 0
    for i in range(len(y_pred)):
        total += (y_pred[i] - v_y[i]) ** 2
    Eout = total / len(y_pred)
    print("Eout: ",Eout)
    # build tree

def exec_10():
    # load data
    print("problem 10")
    train_y, train_x = svm_read_problem("hw6_train.dat.txt")
    test_y, test_x = svm_read_problem("hw6_test.dat.txt")
    num_leaves, num_train, num_f = 1, len(train_y), len(train_x[0])
    # init
    t_x, t_y = list(), list()
    for i in range(num_train):
        t_y.append(train_y[i])
        tmpx = list()
        for j in range(num_f):
            tmpx.append(train_x[i][j + 1])
        t_x.append(tmpx)
    t_x, t_y = np.array(t_x), np.array(t_y)
    v_x, v_y = list(), list()
    for i in range(len(test_y)):
        v_y.append(test_y[i])
        tmpx = list()
        for j in range(num_f):
            tmpx.append(test_x[i][j + 1])
        v_x.append(tmpx)
    v_x, v_y = np.array(v_x), np.array(v_y)
    # build tree
    T = 200
    Forest = list()
    Eouts = list()
    for i in range(T):
        # do some sampling
        print("number of tree: ",i)
        random.seed(i)
        idx_list = random.sample(range(num_train), num_train//2)
        s_x, s_y = t_x[idx_list], t_y[idx_list]
        tree = CARTRegressionTree()
        tree.fit(s_x, s_y)
        Forest.append(tree)
        y_pred = tree.predict(v_x)
        # calculate squared error
        total = 0
        for i in range(len(y_pred)):
            total += (y_pred[i] - v_y[i]) ** 2
        Eout = total / len(y_pred)
        print("Eout: ",Eout)
        Eouts.append(Eout)
    plt.hist(Eouts)
    plt.show()
    # problem 11
    print("problem 11")
    Eins = []
    for i in range(T):
        pred = Forest[i].predict(t_x)
        # calculate squared error
        total = 0
        for i in range(len(pred)):
            total += (pred[i] - t_y[i]) ** 2
        Ein = total / len(pred)
        print("Ein: ",Ein)
        Eins.append(Ein)
    # forest ein eout
    total_in, total_out = 0, 0
    pred_y = np.zeros(len(v_y))
    pred_yin = np.zeros(len(t_y))
    for i in range(T):
        pred = Forest[i].predict(v_x)
        predin = Forest[i].predict(t_x)
        pred_y += pred
        pred_yin += predin
    pred_y = pred_y / T
    pred_yin = pred_yin / T
    for i in range(len(pred_y)):
        total_out += (pred_y[i] - v_y[i]) ** 2
    for i in range(len(pred_yin)):
        total_in += (pred_yin[i] - t_y[i]) ** 2
    average_in, average_out = total_in / len(pred_y), total_out / len(pred_yin)
    plt.scatter(Eins,Eouts,c="b",s = 100)
    plt.scatter([average_in],[average_out],c="r",s = 100)
    plt.show()
    # problem 12
    print("problem 12")
    # first t forest
    Etins, Etouts = [], []
    for i in range(T):
        t = i + 1
        total_in, total_out = 0, 0
        pred_y = np.zeros(len(v_y))
        pred_yin = np.zeros(len(t_y))
        for j in range(t):
            pred = Forest[j].predict(v_x)
            predin = Forest[j].predict(t_x)
            pred_y += pred
            pred_yin += predin
        pred_y = pred_y / t
        pred_yin = pred_yin / t
        total_in, total_out = 0, 0
        for j in range(len(pred_y)):
            total_out += (pred_y[j] - v_y[j]) ** 2
        for j in range(len(pred_yin)):
            total_in += (pred_yin[j] - t_y[j]) ** 2
        average_in, average_out = total_in / len(pred_y), total_out / len(pred_yin)
        Etins.append(average_in)
        Etouts.append(average_out)
    plt.plot(range(T),Etouts,c="b")
    plt.plot(range(T),Eouts,c="r")
    plt.show()


def main():
    # load data
    #exec_9()
    exec_10()


if __name__ == "__main__":
    main()
