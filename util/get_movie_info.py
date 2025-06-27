import os

import pandas as pd
import requests
from urllib.parse import quote_plus
import time

from tabulate import tabulate

OMDB_API = '435e4290'  # OMDb API 密钥
USER_RATING_COLUMN = ['userId', 'movieId', 'rating', 'timestamp']
U_ITEM_COLUMN = ['movieId', 'title', 'release_date', 'video_release_date', 'imdb_url', 'unknown',
                 'action', 'adventure', 'animation', 'children', 'comedy',
                 'crime', 'documentary', 'drama', 'fantasy', 'film_noir',
                 'horror', 'musical', 'mystery', 'romance', 'sci_fi',
                 'thriller', 'war', 'western']
MOVIE_ITEM_COLUMN = ['movieId', 'title', 'release_date', 'video_release_date', 'imdb_url', 'unknown',
                     'action', 'adventure', 'animation', 'children', 'comedy',
                     'crime', 'documentary', 'drama', 'fantasy', 'film_noir',
                     'horror', 'musical', 'mystery', 'romance', 'sci_fi',
                     'thriller', 'war', 'western', 'director', 'actor'],
GENRES_COLUMNS = ['action', 'adventure', 'animation', 'children', 'comedy',
                  'crime', 'documentary', 'drama', 'fantasy', 'film_noir',
                  'horror', 'musical', 'mystery', 'romance', 'sci_fi',
                  'thriller', 'war', 'western']


RATINGS_FILE_PATH = '../data/ml-100k/u1.base'
BASE_MOVIES_FILE_PATH = '../data/ml-100k/u.item'
FORMAT_MOVIES_FILE_PATH = '../data/ml-100k/u1_with_directors_and_actors.item'
TEST_FILE_PATH = '../data/ml-100k/u1u.test'

import re


def clean_movie_title(title):
    """使用正则表达式去除 "(1995)" 或 "The (1995)"""
    cleaned_title = re.sub(r'\s*The?\s*\(\d{4}\)$', '', title)  # 移除结尾的 "The (1995)"
    cleaned_title = re.sub(r'\s*\(\d{4}\)$', '', cleaned_title)  # 移除结尾的 "(1995)"
    return cleaned_title


def format_movies_info(file_path, output_file_path):
    """获取电影导演和演员信息"""
    movies_df = pd.read_csv(file_path, sep='|', header=None, names=U_ITEM_COLUMN, encoding='ISO-8859-1')

    print(tabulate(movies_df.head(10), headers='keys', tablefmt='pretty', showindex=False))

    # OMDb API 请求函数，获取电影导演和演员信息
    def get_movie_director_actor(movie_title):
        # 去掉标题中的年份部分
        title_cleaned = clean_movie_title(movie_title)
        title_encoded = quote_plus(title_cleaned)  # 编码清理后的标题
        url = f"http://www.omdbapi.com/?t={title_encoded}&apikey={OMDB_API}"

        try:
            response = requests.get(url, timeout=10)  # 设置请求超时
            response.raise_for_status()  # 如果请求失败，抛出异常
            data = response.json()

            if data.get('Response') == 'True':
                director = data.get('Director', 'Unknown')
                actors = data.get('Actors', 'Unknown')
                return director, actors
            else:
                print(f"无法获取 {movie_title} 的信息: {data.get('Error', 'Unknown Error')}")
                return None, None
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {movie_title} | 错误: {e}")
            return None, None

    movie_info_dict = {}
    # 遍历电影数据，获取导演和主演
    for idx, row in movies_df.iterrows():
        movie_id = row['movieId']
        title = row['title']

        # 获取导演和主演
        director, actors = get_movie_director_actor(title)
        movie_info_dict[movie_id] = {'Director': director, 'Actors': actors}

        # 为了避免过快请求，添加延迟
        time.sleep(1)

    movies_df['Director'] = movies_df['movieId'].map(lambda x: movie_info_dict.get(x, {}).get('Director', 'Unknown'))
    movies_df['Actors'] = movies_df['movieId'].map(lambda x: movie_info_dict.get(x, {}).get('Actors', 'Unknown'))

    # 输出结果到文件
    movies_df.to_csv(output_file_path, sep='|', index=False, header=False)

    print(f"数据已保存到 {output_file_path}")


if __name__ == '__main__':
    # 获取电影导演,演员信息并保存
    format_movies_info(BASE_MOVIES_FILE_PATH, FORMAT_MOVIES_FILE_PATH)
