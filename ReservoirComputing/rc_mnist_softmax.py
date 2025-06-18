import numpy as np
from keras.datasets import mnist
from scipy import sparse
from numpy.linalg import eigvals

# 1) Load & preprocess MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
# we'll use row-by-row encoding: T=28 time steps of 28-dim vectors
T, input_dim = 28, 28

# 2) Reservoir hyperparameters
N = 500         # reservoir size
alpha = 0.3     # leak rate
input_scale = 1.0
rho = 0.95      # spectral radius
reg = 1e-3      # L2 penalty on softmax weights

rng = np.random.RandomState(42)
W_in = (rng.rand(N, input_dim) - 0.5) * 2 * input_scale

# sparse recurrent weights
density = 0.05
W = sparse.rand(N, N, density=density, random_state=rng).A
W[W != 0] -= 0.5
W *= rho / max(abs(eigvals(W)))

def reservoir_states(X):
    """Run each image through the reservoir and collect final states."""
    M = X.shape[0]
    H = np.zeros((M, N))
    for i in range(M):
        x = np.zeros(N)
        for t in range(T):
            u = X[i, t]               # shape=(28,)
            pre = W.dot(x) + W_in.dot(u)
            x = (1 - alpha) * x + alpha * np.tanh(pre)
        H[i] = x
    return H

# 3) Build reservoir features
H_train = reservoir_states(X_train)
H_test  = reservoir_states(X_test)

# 4) Softmax Read-out
class SoftmaxReadout:
    def __init__(self, lr=0.5, epochs=200, reg_lambda=1e-3, verbose=False):
        self.lr = lr
        self.epochs = epochs
        self.reg = reg_lambda
        self.verbose = verbose

    def _softmax(self, Z):
        Z = Z - Z.max(axis=1, keepdims=True)      # for numerical stability
        expZ = np.exp(Z)
        return expZ / expZ.sum(axis=1, keepdims=True)

    def _one_hot(self, y, C):
        Y = np.zeros((y.size, C))
        Y[np.arange(y.size), y] = 1
        return Y

    def fit(self, H, y):
        M, D = H.shape
        C = y.max() + 1
        Y = self._one_hot(y, C)
        self.W = np.zeros((D, C))

        for epoch in range(1, self.epochs + 1):
            scores = H.dot(self.W)                # shape=(M,C)
            P = self._softmax(scores)            # shape=(M,C)

            # cross-entropy + L2 penalty
            loss = -np.mean(np.sum(Y * np.log(P + 1e-15), axis=1)) \
                   + 0.5 * self.reg * np.sum(self.W**2)

            # gradient
            grad = (H.T.dot(P - Y)) / M
            grad += self.reg * self.W

            # update
            self.W -= self.lr * grad

            if self.verbose and epoch % 20 == 0:
                print(f"Epoch {epoch:3d} — loss: {loss:.4f}")

        return self

    def predict_proba(self, H):
        return self._softmax(H.dot(self.W))

    def predict(self, H):
        return np.argmax(self.predict_proba(H), axis=1)


# 5) Train & evaluate
readout = SoftmaxReadout(lr=0.5, epochs=200, reg_lambda=reg, verbose=True)
readout.fit(H_train, y_train)

y_pred = readout.predict(H_test)
acc = np.mean(y_pred == y_test)
print(f"\nSoftmax read-out accuracy on MNIST: {acc*100:.2f}%")