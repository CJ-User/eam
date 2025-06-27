import numpy as np
import pandas as pd


# 1. 数据加载与预处理
def load_data(file_path):
    columns = ["user_id", "item_id", "rating", "timestamp"]
    df = pd.read_csv(file_path, sep='\t', names=columns)

    print(f"数据集加载完成: {len(df)}条评分记录")
    print(f"用户数量: {df['user_id'].nunique()}, 电影数量: {df['item_id'].nunique()}")

    return df

# 2. 评分矩阵构建
def build_rating_matrix(df):
    n_users = df['user_id'].nunique()
    n_items = df['item_id'].nunique()

    # 创建从原始ID到矩阵索引的映射
    user_mapping = {user: idx for idx, user in enumerate(df['user_id'].unique())}
    item_mapping = {item: idx for idx, item in enumerate(df['item_id'].unique())}

    # 构建评分矩阵
    R = np.zeros((n_users, n_items))
    for _, row in df.iterrows():
        user_idx = user_mapping[row['user_id']]
        item_idx = item_mapping[row['item_id']]
        R[user_idx, item_idx] = row['rating']

    sparsity = 100 * (1 - np.count_nonzero(R) / (n_users * n_items))

    print(f"评分矩阵构建完成 | 维度: {R.shape} | 稀疏度: {sparsity:.2f}%")

    return R, user_mapping, item_mapping


import pandas as pd


def load_movie_attributes(file_path="../data/ml-100k/u.item"):
    # 完整的24个列名（包含5个元数据列+19个类型列）
    full_columns = [
        "movie_id", "title", "release_date", "video_release_date", "imdb_url",
        "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film_Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]

    # 读取文件并指定所有列名
    movies = pd.read_csv(
        file_path,
        sep="|",
        encoding="latin-1",
        header=None,
        names=full_columns
    )

    # 仅提取19个类型列（第5列到最后）
    item_attrs = movies.iloc[:, 5:].values.astype(np.float32)
    return item_attrs