from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn import svm
import pandas as pd
import glob
import jieba
import spacy
from spacy.lang.zh.stop_words import STOP_WORDS
import numpy as np
from collections import Counter
import seaborn as sn
import matplotlib.pyplot as plt
import pickle

class Model:
    # read all csv to label and combine csv files
    def labeling(self, csvPath, label_mapping, allNewsDataset):
        li = []
        for data in glob.glob(csvPath):
            df = pd.read_csv(data, encoding='utf-8-sig')
            df['target'] = df['label'].map(label_mapping)
            li.append(df)
        # 過濾content為nan的值
        pd.concat(li, axis=0, ignore_index=True).dropna(subset=['target'])\
                .to_csv(allNewsDataset, encoding='utf-8-sig',index=0)

    def DataVisualize(self, df):
        #避免中文亂碼
        plt.rcParams['font.sans-serif']=['SimSun'] 
        plt.rcParams['axes.unicode_minus']=False 

        labelcount = df['label'].value_counts(sort = False)
        labelcount.plot(kind="bar")
        plt.xlabel("類別")
        plt.ylabel("文章數量")
        plt.title("新聞文章")
        plt.show()

    # title and content做斷詞，並寫入原本的csv檔案
    def ArticleCorpus(self, df, allNewsDataset):
        titles, contents = df['title'], df['content']
        # 載入中文字典
        jieba.load_userdict('dict_taiwan.txt')
        
        # stop words
        
        with open('stop_word.txt','r',encoding='utf-8') as f:
            for line in f:
                stop = f.read().splitlines()

        for idx, content in enumerate(contents):
            content_corpus = jieba.cut(content)
            content_join =  ''
            for c in content_corpus:
                #過濾停用字詞
                if c not in stop:
                    content_join += ' ' + str(c)
            df.loc[idx,'content_corpus'] = content_join
        
        for idx, title in enumerate(titles):
            title_corpus = jieba.cut(title)
            title_join = ''
            for t in title_corpus:
                #過濾停用字詞
                if t not in STOP_WORDS:
                    title_join += ' ' + str(t)
            # 寫入 title corpus欄位
            df.loc[idx,'title_corpus'] = title_join
        df.to_csv(allNewsDataset, index=0, encoding='utf-8-sig')

    # 特徵抽取
    def countVec(self, trainWords):
        count_vect = CountVectorizer(analyzer='word',token_pattern=u"(?u)\\b\\w+\\b", max_features=25)
        trainX =  count_vect.fit_transform(trainWords)
        #print(trainX)
        trainX = trainX.toarray()
        return trainX
    
    def TFVec(self, trainWords):
        tf_vect = TfidfVectorizer(analyzer='word',token_pattern=u"(?u)\\b\\w+\\b", max_features=32, min_df=2)
        trainX =  tf_vect.fit_transform(trainWords)
        print(trainX)
        trainX = trainX.toarray()
        return trainX

    def buildModel(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
        model = RandomForestClassifier(n_estimators=2200, max_features="auto", max_depth=13, criterion='gini')
        #model = svm.SVC(kernel='rbf', degree=3, gamma='auto', C=0.5, probability=True)
        model.fit(X_train, y_train)
        pickle.dump(model, open('model.h5', 'wb'))
        self.predict(model, X_test, y_test)

    def predict(self, model, X_test, y_test):
        pred = model.predict(X_test)
        acc = model.score(X_test, y_test)
        print(pred)
        print("accuracy = ", acc)
        self.confusionMatrix(y_test, pred)

    def confusionMatrix(self, y_test, pred):
        print(confusion_matrix(y_test, pred))
        df = pd.DataFrame(confusion_matrix(y_test, pred), range(4), range(4))
        sn.set(font_scale=1.4) # for label size
        sn.heatmap(df, annot=True, cmap='Blues', fmt='g') # font size
        plt.show()
