import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from util.data_loader import load_data, build_rating_matrix




def calculate_rmse(model, test_df, user_mapping, item_mapping):
    """计算测试集的RMSE"""
    y_true = []
    y_pred = []

    for idx, row in test_df.iterrows():
        user_id = row['user_id']  # 根据实际列名调整
        item_id = row['item_id']  # 根据实际列名调整
        rating = row['rating']  # 根据实际列名调整

        # 获取映射后的索引
        u_idx = user_mapping.get(user_id)
        i_idx = item_mapping.get(item_id)

        # 只处理训练集中存在的用户和物品
        if u_idx is not None and i_idx is not None:
            pred = model.predict(u_idx, i_idx)
            print("y_true:",rating, "y_pred:",pred)

            y_true.append(rating)
            y_pred.append(pred)

    # 计算RMSE
    mse = mean_squared_error(y_true, y_pred)
    print()
    return np.sqrt(mse)

class MatrixFactorization:
    def __init__(self, R, k=2, alpha=0.01, beta=0.1, epochs=20):
        """
        :param R: 用户-物品评分矩阵
        :param k: 隐因子维度
        :param alpha: 学习率
        :param beta: 正则化系数
        :param epochs: 迭代次数
        """
        self.R = R
        self.num_users, self.num_items = R.shape
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.epochs = epochs

        # 初始化用户和物品隐因子矩阵
        self.P = np.random.normal(scale=1. / k, size=(self.num_users, k))
        self.Q = np.random.normal(scale=1. / k, size=(self.num_items, k))

        # 获取已知评分的索引
        self.user_indices, self.item_indices = np.where(R > 0)
        self.samples = list(zip(self.user_indices, self.item_indices))
        print(f"模型初始化完成 | 隐因子: {k} | 正则化: {beta} | 学习率: {alpha}")

    def train(self):
        for epoch in range(self.epochs):
            # 随机打乱样本顺序
            # np.random.shuffle(self.samples)

            for u, i in self.samples:
                # 计算预测误差
                r_ui = self.R[u, i]
                r_hat = np.dot(self.P[u], self.Q[i])
                e_ui = r_ui - r_hat

                # 更新参数
                self.Q[i] += self.alpha * (e_ui * self.P[u] - self.beta * self.Q[i])
                self.P[u] += self.alpha * (e_ui * self.Q[i] - self.beta * self.P[u])

            # 动态衰减学习率
            # if (epoch + 1) % 20 == 0:
            #     self.alpha *= 0.9

            # 简化的进度输出
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{self.epochs} completed")

    def predict(self, user_idx, item_idx):
        """预测用户对物品的评分"""
        return np.dot(self.P[user_idx], self.Q[item_idx])


if __name__ == "__main__":
    # 配置参数
    DATA_PATH = "../data/ml-100k/u.data"  # 根据实际路径调整

    # 加载并预处理数据
    df = load_data(DATA_PATH)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 构建评分矩阵
    R_train, user_mapping, item_mapping = build_rating_matrix(train_df)

    # 初始化模型
    model = MatrixFactorization(R_train)

    # 训练模型
    print("\n开始训练矩阵分解模型...")
    model.train()

    # 计算测试集RMSE
    rmse = calculate_rmse(model, test_df, user_mapping, item_mapping)
    print(f"\n测试集RMSE: {rmse:.4f}")

