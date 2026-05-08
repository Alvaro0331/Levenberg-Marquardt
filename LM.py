import numpy as np


# ---------------------------------------------------------------------------
# Funciones de activación
# ---------------------------------------------------------------------------

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1.0 - s)


# ---------------------------------------------------------------------------
# Red neuronal multicapa con entrenamiento Levenberg-Marquardt
# ---------------------------------------------------------------------------

class MLP:
    def __init__(self, layer_sizes):
        """
        layer_sizes: lista de enteros, ej. [2, 4, 1]
                     [n_entradas, n_ocultas, n_salidas]
        """
        self.layer_sizes = layer_sizes
        n_in, n_hidden, n_out = layer_sizes

        # Inicialización Xavier
        self.W1 = np.random.randn(n_in, n_hidden) * np.sqrt(1.0 / n_in)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_out) * np.sqrt(1.0 / n_hidden)
        self.b2 = np.zeros((1, n_out))

    # ------------------------------------------------------------------
    # Utilidades de aplanamiento / reconstrucción
    # ------------------------------------------------------------------

    def flatten_weights(self):
        """Concatena todos los parámetros en un vector 1D."""
        return np.concatenate([
            self.W1.ravel(), self.b1.ravel(),
            self.W2.ravel(), self.b2.ravel()
        ])

    def unflatten_weights(self, w):
        """Reconstruye las matrices de pesos desde un vector 1D."""
        n_in, n_hidden, n_out = self.layer_sizes
        off = 0
        W1 = w[off:off + n_in * n_hidden].reshape(n_in, n_hidden)
        off += n_in * n_hidden
        b1 = w[off:off + n_hidden].reshape(1, n_hidden)
        off += n_hidden
        W2 = w[off:off + n_hidden * n_out].reshape(n_hidden, n_out)
        off += n_hidden * n_out
        b2 = w[off:off + n_out].reshape(1, n_out)
        return W1, b1, W2, b2

    # ------------------------------------------------------------------
    # Paso hacia adelante
    # ------------------------------------------------------------------

    def forward(self, X, w=None):
        """
        Propagación hacia adelante.
        Si se pasa w, usa esos pesos (sin modificar self).
        Devuelve: A2, Z1, A1, Z2
        """
        if w is not None:
            W1, b1, W2, b2 = self.unflatten_weights(w)
        else:
            W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2

        Z1 = X @ W1 + b1          # (m, n_hidden)
        A1 = sigmoid(Z1)           # (m, n_hidden)
        Z2 = A1 @ W2 + b2          # (m, n_out)
        A2 = sigmoid(Z2)           # (m, n_out)
        return A2, Z1, A1, Z2

    # ------------------------------------------------------------------
    # Residuos  e = ŷ - y  (vector columna aplanado)
    # ------------------------------------------------------------------

    def compute_residuals(self, X, y, w=None):
        """e de shape (m * n_out,)"""
        A2, _, _, _ = self.forward(X, w)
        return (A2 - y).ravel()

    # ------------------------------------------------------------------
    # Jacobiano  J  de shape (m * n_out, n_weights)
    # ------------------------------------------------------------------

    def compute_jacobian(self, X, y):
        """
        Calcula J mediante retropropagación de sensibilidades.

        Para la muestra i y la salida k:
          δ²_k  = σ'(z²_{i,k})
          δ¹_k  = δ²_k · W2[:,k] ⊙ σ'(z¹_{i,:})

          ∂e_{i,k}/∂W1[p,j] = δ¹_k[j] · x_{i,p}
          ∂e_{i,k}/∂b1[j]   = δ¹_k[j]
          ∂e_{i,k}/∂W2[j,k] = δ²_k · A1[i,j]  (cero para columnas l≠k)
          ∂e_{i,k}/∂b2[k]   = δ²_k             (cero para l≠k)
        """
        m = X.shape[0]
        n_in, n_hidden, n_out = self.layer_sizes
        n_weights = n_in * n_hidden + n_hidden + n_hidden * n_out + n_out

        _, Z1, A1, Z2 = self.forward(X)
        sp1 = sigmoid_prime(Z1)   # (m, n_hidden)
        sp2 = sigmoid_prime(Z2)   # (m, n_out)

        J = np.zeros((m * n_out, n_weights))

        off_W1 = 0
        off_b1 = n_in * n_hidden
        off_W2 = off_b1 + n_hidden
        off_b2 = off_W2 + n_hidden * n_out

        for i in range(m):
            for k in range(n_out):
                row = i * n_out + k
                d2 = sp2[i, k]                              # escalar
                d1 = d2 * self.W2[:, k] * sp1[i, :]        # (n_hidden,)

                # ∂e/∂W1  → producto exterior aplanado
                J[row, off_W1:off_b1] = np.outer(X[i], d1).ravel()
                # ∂e/∂b1
                J[row, off_b1:off_W2] = d1
                # ∂e/∂W2  → solo la columna k de W2 (stride n_out, limitado a n_hidden)
                idx_W2 = off_W2 + k + np.arange(n_hidden) * n_out
                J[row, idx_W2] = d2 * A1[i, :]
                # ∂e/∂b2  → solo la posición k
                J[row, off_b2 + k] = d2

        return J

    # ------------------------------------------------------------------
    # Entrenamiento con Levenberg-Marquardt
    # ------------------------------------------------------------------

    def train(self, X, y, max_iter=1000, lam=0.01,
              lam_inc=10.0, lam_dec=0.1,
              tol_grad=1e-6, tol_loss=1e-10, verbose=True):
        """
        Parámetros
        ----------
        lam      : factor de amortiguamiento inicial λ
        lam_inc  : factor para aumentar λ cuando se rechaza el update
        lam_dec  : factor para disminuir λ cuando se acepta el update
        tol_grad : parada si ‖Jᵀe‖ < tol_grad
        tol_loss : parada si |ΔMSE| < tol_loss
        """
        self.history = []
        w = self.flatten_weights()

        e = self.compute_residuals(X, y, w)
        loss = np.mean(e ** 2)

        for it in range(max_iter):
            # Sincronizar self con el vector w actual (necesario para Jacobiano)
            self.W1, self.b1, self.W2, self.b2 = self.unflatten_weights(w)

            J = self.compute_jacobian(X, y)
            e = self.compute_residuals(X, y, w)
            loss = np.mean(e ** 2)

            JtJ = J.T @ J          # Hessiano aproximado  (n_w, n_w)
            Jte = J.T @ e          # Gradiente            (n_w,)

            # Criterio de parada por gradiente
            if np.linalg.norm(Jte) < tol_grad:
                if verbose:
                    print(f"Convergencia por gradiente en iteración {it}.")
                break

            # Resolver el sistema (JᵀJ + λI) Δw = -Jᵀe
            n_w = len(w)
            try:
                delta_w = -np.linalg.solve(JtJ + lam * np.eye(n_w), Jte)
            except np.linalg.LinAlgError:
                lam *= lam_inc
                continue

            w_new = w + delta_w
            e_new = self.compute_residuals(X, y, w_new)
            loss_new = np.mean(e_new ** 2)

            if loss_new < loss:
                # Update aceptado
                loss_prev = loss
                w = w_new
                loss = loss_new
                lam = max(lam * lam_dec, 1e-12)

                if abs(loss_prev - loss) < tol_loss:
                    if verbose:
                        print(f"Convergencia por cambio en pérdida en iteración {it}.")
                    break
            else:
                # Update rechazado: aumentar amortiguamiento
                lam = min(lam * lam_inc, 1e12)

            if verbose and it % 100 == 0:
                print(f"Iter {it:5d} | MSE: {loss:.8f} | λ: {lam:.2e}")

            self.history.append({'iter': it, 'loss': loss, 'lambda': lam})

        # Guardar pesos finales en self
        self.W1, self.b1, self.W2, self.b2 = self.unflatten_weights(w)
        return self.history

    # ------------------------------------------------------------------
    # Predicción
    # ------------------------------------------------------------------

    def predict(self, X):
        A2, _, _, _ = self.forward(X)
        return A2


# ---------------------------------------------------------------------------
# Bloque de prueba — XOR
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    nn = MLP(layer_sizes=[2, 4, 1])
    nn.train(X, y, max_iter=500, lam=0.01, verbose=True)

    preds = nn.predict(X)
    print("\nPredicciones finales:")
    for xi, yi, pi in zip(X, y, preds):
        print(f"  x={xi}  y_true={yi[0]}  y_pred={pi[0]:.4f}")
