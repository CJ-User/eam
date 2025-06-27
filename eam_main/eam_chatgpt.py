import numpy as np

# ====== 模型超参数 ======
latent_dim = 10
lr = 0.001
reg = 0.01
epochs = 10
clip_norm = 1.0

# ====== 数据集大小参数 ======
num_users = 943
num_items = 1682
num_genres = 19  # genre 是 one-hot 19维

# ====== 数据加载函数 ======
def load_data():
    ratings = []
    with open('../data/ml-100k/u.data', 'r') as f:
        for line in f:
            user, item, rating, _ = map(int, line.strip().split('\t'))
            ratings.append((user - 1, item - 1, rating))
    ratings = np.array(ratings)

    item_genres = np.zeros((num_items, num_genres))
    with open('../data/ml-100k/u.item', encoding='latin1') as f:
        for i, line in enumerate(f):
            fields = line.strip().split('|')
            genre_info = list(map(int, fields[-19:]))
            item_genres[i] = genre_info

    return ratings, item_genres

# ====== 梯度裁剪函数 ======
def clip_grad(grad, max_norm=1.0):
    norm = np.linalg.norm(grad)
    if norm > max_norm:
        return grad * (max_norm / norm)
    return grad

# ====== 评分预测函数 ======
def predict(u, j, U, F, omega, theta, item_genres):
    genres = item_genres[j]
    indices = np.where(genres == 1)[0]
    if len(indices) == 0:
        return 3.0
    total = 0
    for k in indices:
        total += omega[u, k] * theta[j, k] * np.dot(U[u], F[k])
    return total / len(indices)

# ====== 训练主函数 ======
def train():
    ratings, item_genres = load_data()

    # 参数初始化（较小值）
    U = np.random.normal(0, 0.01, (num_users, latent_dim))
    F = np.random.normal(0, 0.01, (num_genres, latent_dim))
    omega = np.random.normal(0.01, 0.001, (num_users, num_genres))
    theta = np.random.normal(0.01, 0.001, (num_items, num_genres))

    for epoch in range(1, epochs + 1):
        np.random.shuffle(ratings)
        total_loss = 0

        for u, j, r in ratings:
            genre_vec = item_genres[j]
            indices = np.where(genre_vec == 1)[0]
            if len(indices) == 0:
                continue

            pred = predict(u, j, U, F, omega, theta, item_genres)
            err = r - pred
            total_loss += err ** 2

            for k in indices:
                dot = np.dot(U[u], F[k])
                grad = -2 * err / len(indices)

                # 梯度裁剪更新
                grad_U = grad * omega[u, k] * theta[j, k] * F[k] + reg * U[u]
                grad_F = grad * omega[u, k] * theta[j, k] * U[u] + reg * F[k]
                grad_omega = grad * theta[j, k] * dot + reg * omega[u, k]
                grad_theta = grad * omega[u, k] * dot + reg * theta[j, k]

                U[u] -= lr * clip_grad(grad_U, clip_norm)
                F[k] -= lr * clip_grad(grad_F, clip_norm)
                omega[u, k] -= lr * clip_grad(grad_omega, clip_norm)
                theta[j, k] -= lr * clip_grad(grad_theta, clip_norm)

        rmse = np.sqrt(total_loss / len(ratings))
        print(f"Epoch {epoch}/{epochs}, RMSE: {rmse:.4f}")

# ====== 执行 ======
if __name__ == "__main__":
    train()
