import numpy as np

class CARTRegressionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def mse(self, y):
        return np.mean((y - np.mean(y))**2)

    def split(self, X, y, feature_idx, threshold):
        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    def find_best_split(self, X, y):
        best_mse = np.inf
        best_feature_idx = None
        best_threshold = None

        for feature_idx in range(X.shape[1]):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self.split(X, y, feature_idx, threshold)
                mse = self.mse(y_left) + self.mse(y_right)
                if mse < best_mse:
                    best_mse = mse
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold

    def build_tree(self, X, y, depth):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return np.mean(y)

        feature_idx, threshold = self.find_best_split(X, y)
        if feature_idx is None or threshold is None:
            return np.mean(y)

        X_left, y_left, X_right, y_right = self.split(X, y, feature_idx, threshold)

        left_subtree = self.build_tree(X_left, y_left, depth+1)
        right_subtree = self.build_tree(X_right, y_right, depth+1)

        return {'feature_idx': feature_idx, 'threshold': threshold,
                'left': left_subtree, 'right': right_subtree}

    def fit(self, X, y):
        self.tree = self.build_tree(X, y, depth=0)

    def predict_single(self, x, node):
        if isinstance(node, dict):
            if x[node['feature_idx']] <= node['threshold']:
                return self.predict_single(x, node['left'])
            else:
                return self.predict_single(x, node['right'])
        else:
            return node

    def predict(self, X):
        return np.array([self.predict_single(x, self.tree) for x in X])