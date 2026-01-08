import numpy as np
import scipy.sparse as sp
from numpy.linalg import inv, norm


def soft_threshold(X, thresh):
    return np.sign(X) * np.maximum(np.abs(X) - thresh, 0.0)


class ADMMSLIM:
    """
    ADMM-SLIM (WSDM 2020)
    Exact paper-consistent implementation
    """

    def __init__(
        self,
        l1=1.0,
        l2=0.0,
        rho=1.0,
        max_iter=50,
        tol=1e-4,
    ):
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.rho = float(rho)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.W = None  # item-item matrix

    def fit(self, X: sp.csr_matrix):
        if not sp.isspmatrix_csr(X):
            raise ValueError("X must be csr_matrix")

        _, I = X.shape
        XtX = (X.T @ X).toarray().astype(np.float32)

        eye = np.eye(I, dtype=np.float32)

        # ADMM variables (all I x I)
        W = np.zeros((I, I), dtype=np.float32)
        Z = np.zeros((I, I), dtype=np.float32)
        U = np.zeros((I, I), dtype=np.float32)

        # Precompute inverse
        A = XtX + (self.l2 + self.rho) * eye
        A_inv = inv(A).astype(np.float32)

        for _ in range(self.max_iter):
            W_old = W.copy()

            # W-update
            W = A_inv @ (XtX + self.rho * (Z - U))
            np.fill_diagonal(W, 0.0)

            # Z-update (pure soft-threshold)
            Z = soft_threshold(W + U, self.l1 / self.rho)
            np.fill_diagonal(Z, 0.0)

            # Dual update
            U += W - Z

            # ADMM stopping criterion (primal residual)
            if norm(W - W_old, ord="fro") < self.tol:
                break

        self.W = Z  # final sparse solution

    def predict(self, X):
        if self.W is None:
            raise RuntimeError("Call fit() first")
        return X @ self.W