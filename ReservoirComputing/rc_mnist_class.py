import numpy as np
from keras.datasets import mnist
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
from numpy.linalg import eigvals

# 1) Load & preprocess MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Normalize to [0,1]
X_train, X_test = X_train / 255.0, X_test / 255.0

# We'll treat each image as 28 time steps of 28-dim input
T, in_dim = 28, 28

# 2) Reservoir hyperparameters
N = 500           # reservoir size
alpha = 0.3       # leak rate
input_scale = 1.0 # input weight scale
rho = 0.95        # desired spectral radius
reg = 1e-2        # ridge regression strength

# 3) Initialize weights
rng = np.random.RandomState(42)
W_in = (rng.rand(N, in_dim) - 0.5) * 2 * input_scale
# sparse random W
density = 0.05
W = sparse.rand(N, N, density=density, random_state=rng).A
W[W != 0] -= 0.5
# scale to spectral radius rho
sr = max(abs(eigvals(W)))
W *= (rho / sr)

# 4) Helper: run reservoir on one batch of images
def reservoir_states(X):
    # X shape = (M, 28, 28)
    M = X.shape[0]
    H = np.zeros((M, N))
    for i in range(M):
        x = np.zeros(N)
        for t in range(T):
            u = X[i, t]            # shape (28,)
            pre = W.dot(x) + W_in.dot(u)
            x = (1 - alpha) * x + alpha * np.tanh(pre)
        H[i] = x
    return H

# 5) Compute reservoir states
H_train = reservoir_states(X_train)
H_test  = reservoir_states(X_test)

# 6) Train read-out (one-versus-rest ridge classifier)
clf = RidgeClassifier(alpha=reg, fit_intercept=False)
clf.fit(H_train, y_train)

# 7) Evaluate
acc = clf.score(H_test, y_test)
print(f"MNIST classification accuracy with ESN read-out: {acc*100:.2f}%")