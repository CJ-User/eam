import numpy as np

class MMF:
    def __init__(self, R, item_attrs, k=40, alpha=0.02, beta=0.005, epochs=150):
        self.R = R
        self.item_attrs = item_attrs
        self.num_users, self.num_items = R.shape
        self.num_genres = item_attrs.shape[1]
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.epochs = epochs

        self.U = np.random.normal(scale=1. / k, size=(self.num_users, k))
        self.F = np.random.normal(scale=1. / k, size=(self.num_genres, k))

        self.omega = np.zeros((self.num_users, self.num_genres), dtype=np.float32)
        for g in range(self.num_genres):
            if np.any(item_attrs[:, g] > 0):
                self.omega[:, g] = np.random.uniform(0.1, 0.3, size=self.num_users)

        self.theta = np.zeros_like(item_attrs, dtype=np.float32)
        non_zero_mask = item_attrs > 0
        self.theta[non_zero_mask] = np.random.uniform(0.05, 0.15, size=np.sum(non_zero_mask))

        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.mu = np.mean(R[R > 0])

        self.user_indices, self.item_indices = np.where(R > 0)
        self.samples = list(zip(self.user_indices, self.item_indices))

    def predict(self, u, i):
        genres = np.where(self.item_attrs[i] > 0)[0]
        total = 0.0
        for g in genres:
            dot_val = np.dot(self.U[u], self.F[g])
            dot_val = np.clip(dot_val, -10, 10)
            total += self.omega[u, g] * self.theta[i, g] * dot_val

        pred = self.mu + self.b_u[u] + self.b_i[i] + total / (len(genres) + 1e-8)
        return np.clip(pred, 1, 5)

    def train(self, verbose=True):
        for epoch in range(self.epochs):
            if epoch < 50:
                current_alpha = self.alpha
            else:
                current_alpha = self.alpha * (0.9 ** ((epoch - 50) // 10))

            np.random.shuffle(self.samples)
            total_loss = 0.0

            for idx, (u, i) in enumerate(self.samples):
                r_ui = self.R[u, i]
                r_hat = self.predict(u, i)
                e_ui = r_ui - r_hat
                total_loss += e_ui ** 2

                genres = np.where(self.item_attrs[i] > 0)[0]
                for g in genres:
                    dot = np.dot(self.U[u], self.F[g])
                    dot = np.clip(dot, -10, 10)

                    grad_U = e_ui * self.omega[u, g] * self.theta[i, g] * self.F[g]
                    grad_F = e_ui * self.omega[u, g] * self.theta[i, g] * self.U[u]
                    grad_omega = e_ui * self.theta[i, g] * dot
                    grad_theta = e_ui * self.omega[u, g] * dot

                    for grad in [grad_U, grad_F]:
                        norm = np.linalg.norm(grad)
                        if norm > 1.0:
                            grad /= norm

                    self.U[u] += current_alpha * (grad_U - self.beta * self.U[u])
                    self.F[g] += current_alpha * (grad_F - self.beta * self.F[g])

                    new_omega = self.omega[u, g] + current_alpha * (grad_omega - self.beta * self.omega[u, g])
                    new_theta = self.theta[i, g] + current_alpha * (grad_theta - self.beta * self.theta[i, g])
                    self.omega[u, g] = np.clip(new_omega, 0.05, 4.0)
                    self.theta[i, g] = np.clip(new_theta, 0.05, 4.0)

                self.b_u[u] += current_alpha * (e_ui - self.beta * self.b_u[u])
                self.b_i[i] += current_alpha * (e_ui - self.beta * self.b_i[i])

            avg_loss = total_loss / len(self.samples)
            rmse = np.sqrt(avg_loss)

            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"[Epoch {epoch + 1}/{self.epochs}] RMSE: {rmse:.4f} | AvgLoss: {avg_loss:.4f} | LR: {current_alpha:.5f} | Samples: {len(self.samples)}")
                print(f"[Debug] θ mean: {np.mean(self.theta):.4f}, max: {np.max(self.theta):.4f} | ω mean: {np.mean(self.omega):.4f}, max: {np.max(self.omega):.4f}")

        print("\n[Final Params Summary]")
        print(f"U mean: {np.mean(self.U):.4f}, F mean: {np.mean(self.F):.4f}")
        print(f"theta mean: {np.mean(self.theta):.4f}, omega mean: {np.mean(self.omega):.4f}")

    def evaluate_rmse(self):
        mse = 0.0
        for u, i in self.samples:
            r_ui = self.R[u, i]
            r_hat = self.predict(u, i)
            mse += (r_ui - r_hat) ** 2
        return np.sqrt(mse / len(self.samples))
