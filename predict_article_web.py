from flask import Flask, render_template, request
import pickle
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

app = Flask(__name__, template_folder='template',static_folder = 'static',static_url_path='/static')
            
# stop words
with open('stop_word.txt','r',encoding='utf-8') as f:
    for line in f:
        stop = f.read().splitlines() 

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        News_title = str(request.form.get('title'))
        News_content = str(request.form.get('content'))
        
        # Text preprocessing
        model = pickle.load(open('model.h5', 'rb'))
        label_mapping = {0:'政治', 1:'生活', 2:'財經', 
                        3:'健康'}
        # 載入中文字典
        jieba.load_userdict('dict_taiwan.txt')
        arr = text_preprocessing(News_title, News_content)
        pred = model.predict(arr)[0]
        print(model.predict_proba(arr))
        print(arr)
        result = label_mapping[pred]
        return render_template('index.html',News_title=News_title,News_content=News_content,result=result)
    else:
        return render_template('index.html')

def text_preprocessing(News_title, News_content):
    article_join = ''
    article_corpus = jieba.cut(News_title + ' ' + News_content)
    for article in article_corpus:
        if article not in stop:
            article_join += ' ' + str(article)
    return TFvect(article_join)

def TFvect(Textdata):
    vect = TfidfVectorizer(analyzer='word',token_pattern=u"(?u)\\b\\w+\\b", max_features=32, min_df=1)
    testdata = vect.fit_transform([Textdata])
    data_pred = testdata.toarray()
    return data_pred

if __name__ == '__main__':
    app.run(debug=True)