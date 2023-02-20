from trainModel import Model
import pandas as pd
M = Model()

# 所有類別的csv檔案路徑
datasets_path = "news_data/EBCNews/*.csv"
# 標記的類別用dict儲存
label_mapping = {'政治': '0','生活': '1', '財經': '2', '健康':'3'}

# 將所有類別的csv檔案結合成一個csv
allNewsDataset = "news_data/EBCNews.csv"

# 將資料做標記
#M.labeling(datasets_path ,label_mapping, allNewsDataset)

# 資料視覺化
df = pd.read_csv(allNewsDataset, encoding='utf-8-sig').dropna()
M.DataVisualize(df)

# 文章斷詞
#M.ArticleCorpus(df, allNewsDataset)

grouped = df.groupby('target')
n_samples = 1000
samples = []
sampled_df = pd.DataFrame()
for name, group in grouped:
    class_size = group.shape[0]
    sample_proportion = n_samples / class_size
    sample = group.sample(frac=sample_proportion, random_state=42, replace=True)
    samples.append(sample)
sampled_df = pd.concat(samples)
# 特徵抽取
content = sampled_df['content'].to_list()
title = sampled_df['title'].to_list()
TrainData = [t + ' ' + c for t, c in zip(title, content)]
X_train = M.TFVec(TrainData)
y_train = sampled_df['target']

# 建立模型
M.buildModel(X_train, y_train)
