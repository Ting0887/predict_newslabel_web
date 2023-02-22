# predict News Article
## How to use it?
* The web application be used to predict political、life、financial、healthy

## code detail
* text corpus
    * Use jieba and stopword
```
def ArticleCorpus(self, df, allNewsDataset):
        # 載入中文字典
        jieba.load_userdict('dict_taiwan.txt')
        
        # stop words
        with open('stop_word.txt','r',encoding='utf-8') as f:
            for line in f:
                stopwords = set([line.strip() for line in f])
```
        df['content_corpus'] = df['content'].apply(lambda x: ' '.join([w for w in jieba.cut(x) if w not in stopwords]))
        df['title_corpus'] = df['title'].apply(lambda x: ' '.join([w for w in jieba.cut(x) if w not in stopwords]))`
* data sampling
```
grouped = df.groupby('target')
n_samples = 1200
samples = []
sampled_df = pd.DataFrame()
for name, group in grouped:
    class_size = group.shape[0]
    sample_proportion = n_samples / class_size
    sample = group.sample(frac=sample_proportion, random_state=42, replace=True)
    samples.append(sample)
sampled_df = pd.concat(samples)
```
* model parameters
    * n_estimators=1000, max_features="auto", min_samples_split = 2, max_depth=30, criterion='gini'


## model performance
* Accuracy = 95.625%, I think it's not so good, some reasons should be considered, if you have some suggestions, you can tell me :)
![](https://i.imgur.com/1i5XG4Y.png)




## raw data
* It is not convenient to provide the original data, if you are lazy to scrape data, you can pm to me in email

## Please keep in mind
* If you want to use my project, please remember cite the reference source
