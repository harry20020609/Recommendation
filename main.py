import csv
from math import sqrt
import pandas as pd
from tqdm import tqdm


class Recommend:
    def __init__(self):
        self.rating_data = pd.read_csv("dataset/ratings.csv")
        self.movie_data = pd.read_csv("dataset/movies.csv")
        self.all_data = pd.merge(self.movie_data, self.rating_data, on='movieId')
        self.dic = {}
        for i in tqdm(range(len(self.all_data))):
            if self.all_data["userId"][i] in self.dic.keys():
                self.dic[self.all_data["userId"][i]][self.all_data["movieId"][i]] = self.all_data["rating"][i]
            else:
                self.dic[self.all_data["userId"][i]] = {self.all_data["movieId"][i]: self.all_data["rating"][i]}

    def cal(self, user1, user2):
        user1_data = self.dic[user1]
        user2_data = self.dic[user2]
        distance = 0
        for key in user1_data.keys():
            if key in user2_data.keys():
                distance += pow(float(user1_data[key]) - float(user2_data[key]), 2)

        return 1 / (1 + sqrt(distance))

    def similar(self, userId):
        res = []
        for userid in self.dic.keys():
            if not userid == userId:
                similar_score = self.cal(userId, userid)
                res.append((userid, similar_score))
        res.sort(key=lambda val: val[1])
        return res[:4]

    def AVG(self, user):
        return self.rating_data[self.rating_data['userId'] == user]['rating'].mean()

    def sss(self, df):
        return df['genres'].split('|')

    def cosDis(self, movie_vec, usr_vec):
        dotProduct = 0
        U_a_ms = 0
        I_a_ms = 0
        for i in range(0, 19):
            dotProduct += movie_vec[i] * usr_vec[i]
            I_a_ms += movie_vec[i] * movie_vec[i]
            U_a_ms += usr_vec[i] * usr_vec[i]
        if dotProduct == 0:
            cosSim = 0
        else:
            cosSim = dotProduct / (sqrt(I_a_ms) * sqrt(U_a_ms))
        return cosSim

    def predict(self, n=2):
        users_result = {}
        for i in tqdm(range(1, 611)):

            top_sim_user = self.similar(i)[0][0]
            # 相似度最高的用户的观影记录
            items = self.dic[top_sim_user]
            recommendations = []
            # 筛选出该用户未观看的电影并添加到列表中
            for item in items.keys():
                if item not in self.dic[i].keys():
                    recommendations.append((item, items[item]))

            self.movie_data['gene'] = self.movie_data.apply(self.sss, axis=1)
            genesSet = set({})
            for index, row in self.movie_data.iterrows():
                for k in list(row['gene']):
                    genesSet.add(k)
            genesList = list(genesSet)

            # 计算用户向量
            avg = self.AVG(i)
            user_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            rated = self.rating_data[self.rating_data['userId'] == i]
            for j in range(0, 19):
                movie_rated = self.movie_data[self.movie_data['movieId'].isin(rated['movieId'])]
                movie_tagged = pd.DataFrame({}, columns={"movieId", "title", "genres", "gene"})
                for index, row in movie_rated.iterrows():
                    if list(row['gene']).count(genesList[j]) > 0:
                        movie_tagged.loc[len(movie_tagged.index)] = row
                    rated_tagged = rated[rated['movieId'].isin(movie_tagged['movieId'])]
                    n = len(rated_tagged.index)
                    if n != 0:
                        score = (rated_tagged['rating'].sum() / n) - avg
                    else:
                        score = - avg
                    user_vector[j] = score
            # 计算电影向量
            recommens = []
            for j in range(0, len(recommendations)):
                movie_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                for k in range(0, 19):
                    if genesList[k] in list(
                            self.movie_data[self.movie_data['movieId'] == recommendations[j][0]]['gene'])[0]:
                        movie_vector[k] = 1
                recommens.append((recommendations[j][0], self.cosDis(movie_vector, user_vector)))

            recommens.sort(key=lambda val: val[1], reverse=True)  # 按照评分排序

            # 返回评分最高的n部电影
            if len(recommendations) < n:
                users_result[i] = [recom[0] for recom in recommens]
            else:
                users_result[i] = [recom[0] for recom in recommens[:2]]

        with open("result.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["userId", "movieId"])
            for user_id in users_result.keys():
                for recommend_id in users_result[user_id]:
                    writer.writerow([user_id, recommend_id])


def main():
    recommend = Recommend()
    recommend.predict()


if __name__ == "__main__":
    main()
