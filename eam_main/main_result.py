import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from eam_main.eam_model import MMF
from eam_main.mf import MatrixFactorization
from util.data_loader import load_data, build_rating_matrix, load_movie_attributes

LATEN_K = 50
ALPHA = 0.01
BETA = 0.01
EPOCHS = 150


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
            # print("y_true:",rating, "y_pred:",pred)

            y_true.append(rating)
            y_pred.append(pred)

    # 计算RMSE
    mse = mean_squared_error(y_true, y_pred)
    print()
    return np.sqrt(mse)


def mf_result(R_train, test_df, user_mapping, item_mapping):
    # 初始化模型
    model = MatrixFactorization(R_train)

    # 训练模型
    print("\n开始训练矩阵分解模型...")
    model.train()

    # 计算测试集RMSE
    rmse = calculate_rmse(model, test_df, user_mapping, item_mapping)
    print(f"\n测试集RMSE: {rmse:.4f}")


# def mmf_result(train_df, test_df):
#     # 2. 构建属性矩阵
#     item_attrs = load_movie_attributes()
#
#     user_mapping = {id: idx for idx, id in enumerate(train_df["user_id"].unique())}
#     item_mapping = {id: idx for idx, id in enumerate(train_df["item_id"].unique())}
#     R_train = np.zeros((len(user_mapping), len(item_mapping)))
#     for _, row in train_df.iterrows():
#         u = user_mapping[row["user_id"]]
#         i = item_mapping[row["item_id"]]
#         R_train[u, i] = row["rating"]
#
#     # 5. 初始化并训练MMF模型
#     model = MMF(R_train, item_attrs, k=LATEN_K, alpha=ALPHA, beta=BETA, epochs=EPOCHS)
#     model.train()
#
#     rmse = calculate_rmse(model, test_df, user_mapping, item_mapping)
#     print(f"\nmmf测试集RMSE: {rmse:.4f}")


def mmf_result(train_df, test_df):
    # Step 1: 读取所有 item 属性
    all_attrs = load_movie_attributes()  # shape = (1682, 19)

    # Step 2: 创建 user 和 item 映射（仅限训练集）
    user_mapping = {id: idx for idx, id in enumerate(train_df["user_id"].unique())}
    item_mapping = {id: idx for idx, id in enumerate(train_df["item_id"].unique())}

    # Step 3: 构建训练评分矩阵 R_train
    R_train = np.zeros((len(user_mapping), len(item_mapping)))
    for _, row in train_df.iterrows():
        u = user_mapping[row["user_id"]]
        i = item_mapping[row["item_id"]]
        R_train[u, i] = row["rating"]

    # Step 4: 对 item_attrs 做映射，过滤训练集中存在的 item
    item_ids_in_train = train_df["item_id"].unique()
    item_id_to_index = {raw_id: idx for idx, raw_id in enumerate(sorted(item_ids_in_train))}
    item_attrs_aligned = np.zeros((len(item_ids_in_train), all_attrs.shape[1]), dtype=np.float32)

    for i, raw_id in enumerate(item_ids_in_train):
        item_attrs_aligned[i] = all_attrs[raw_id - 1]  # 注意：MovieLens item_id 从 1 开始

    # Step 5: 训练模型
    model = MMF(R_train, item_attrs_aligned, k=LATEN_K, alpha=ALPHA, beta=BETA, epochs=EPOCHS)
    model.train()

    # Step 6: 测试评估
    rmse = calculate_rmse(model, test_df, user_mapping, item_mapping)
    print(f"\nmmf测试集RMSE: {rmse:.4f}")


if __name__ == "__main__":
    # 配置参数
    DATA_PATH = "../data/ml-100k/u.data"  # 根据实际路径调整

    # 加载并预处理数据
    df = load_data(DATA_PATH)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 构建评分矩阵
    R_train, user_mapping, item_mapping = build_rating_matrix(train_df)

    # mf 矩阵
    # mf_result(R_train, test_df, user_mapping, item_mapping)

    # mmf 矩阵
    mmf_result(train_df, test_df)
