import numpy as np
import pandas as pd
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
import pickle


### dataset (newsgroup 데이터는 데이터셋에 noise가 많아서 사용하기 전 전처리가 많이 걸릴듯 하여 제외)
economic = pd.read_csv('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/economic_news_article/economic_news_dataset.csv', delimiter=',', encoding='utf-8')
oshumed = pd.read_csv('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/ohsumed/ohsumed_dataset.csv', delimiter=',', encoding='utf-8')
reuter = pd.read_csv('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/reuter/reuters_dataset.csv', delimiter=',', encoding='utf-8')
# newsgroup = pd.read_csv('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/20newsgroup/20newsgroup_dataset.csv', delimiter=',', encoding='utf-8')

economic_corpus = np.array(economic.iloc[:,1])
oshumed_corpus = np.array(oshumed.iloc[:,1])
reuter_corpus = np.array(reuter.iloc[:,1])














#####################################################################################################
# Tf-Idf
#####################################################################################################

### Tokenization
tokenizer = RegexpTokenizer(r'\w+')
def tokenize(data):
    tok_data = list(map(lambda x: tokenizer.tokenize(x.lower()), data))
    return tok_data
tok_economic = tokenize(economic_corpus)
tok_oshumed = tokenize(oshumed_corpus)
tok_reuter = tokenize(reuter_corpus)

### stopwords
en_stop = get_stop_words('en')
def removeStopwords(data):
    i=0
    corpus = []
    for doc in data:
        print(i)
        corpus.append([word for word in doc if word not in en_stop])
        i += 1
    return corpus
economic_stop = removeStopwords(tok_economic)
oshumed_stop = removeStopwords(tok_oshumed)
reuter_stop = removeStopwords(tok_reuter)

### Stemming
# 오류나는 행들이 조금 있음
p_stemmer = PorterStemmer()
def stem(data):
    tmp = []
    for i in range(len(data)):
        if i % 100:
            print(i)
        doc = data[i]
        try:
            tmp.append([p_stemmer.stem(word) for word in doc])
        except IndexError:
            print('IndexError')
            pass
    return tmp
economic_stem = stem(economic_stop) # 2개 차이가 난다. (4871, 5770)이 이상한 데이터
oshumed_stem = stem(oshumed_stop) # 15개 차이
reuter_stem = stem(reuter_stop) # 차이 없음


### 명사 목록(사전)을 미리 선정하기
def extract_NN(document):
    words_gt_3 = [re.findall('[A-z]{3,}', ' '.join(doc)) for doc in document]
    words_gt_3_unique = sum(words_gt_3, []) #unlist
    words_gt_3_NN = [token[0] for token in nltk.pos_tag(words_gt_3_unique) if token[1].find('NN') == 0]
    return words_gt_3_NN
economic_NN = extract_NN(economic_stem)
oshumed_NN = extract_NN(oshumed_stem)
reuter_NN = extract_NN(reuter_stem)

### 명사 목록(사전)을 미리 선정하기 (가장 많이 출현한 상위 100개 단어)
def extract_top_NN(NN, k=100):
    words_count = Counter(NN)
    voca = [token[0] for token in words_count.most_common(k)]
    return voca
economic_voca = extract_top_NN(economic_NN, 100)
oshumed_voca = extract_top_NN(oshumed_NN, 100)
reuter_voca = extract_top_NN(reuter_NN, 100)


### DTM with tfidf
docu_economic = list(map(lambda x: ' '.join(x), economic_stem))
economic_tfidf_dtm_ = TfidfVectorizer(vocabulary=economic_voca).fit(docu_economic)
economic_tfidf_dtm = economic_tfidf_dtm_.transform(docu_economic).toarray()
pickle.dump(economic_tfidf_dtm_, open('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/representation/economic_tf_idf_model_100.pickle', 'wb'))
pickle.dump(economic_tfidf_dtm, open('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/representation/economic_tf_idf_100.pickle', 'wb'))

docu_oshumed = list(map(lambda x: ' '.join(x), oshumed_stem))
oshumed_tfidf_dtm_ = TfidfVectorizer(vocabulary=oshumed_voca).fit(docu_oshumed)
oshumed_tfidf_dtm = oshumed_tfidf_dtm_.transform(docu_oshumed).toarray()
pickle.dump(oshumed_tfidf_dtm_, open('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/representation/oshumed_tf_idf_model_100.pickle', 'wb'))
pickle.dump(oshumed_tfidf_dtm, open('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/representation/oshumed_tf_idf_100.pickle', 'wb'))

docu_reuter = list(map(lambda x: ' '.join(x), reuter_stem))
reuter_tfidf_dtm_ = TfidfVectorizer(vocabulary=reuter_voca).fit(docu_reuter)
reuter_tfidf_dtm = reuter_tfidf_dtm_.transform(docu_reuter).toarray()
pickle.dump(reuter_tfidf_dtm_, open('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/representation/reuter_tf_idf_model_100.pickle', 'wb'))
pickle.dump(reuter_tfidf_dtm, open('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/representation/reuter_tf_idf_100.pickle', 'wb'))













#####################################################################################################
# Topic modeling(LDA)
#####################################################################################################
from gensim import matutils, models

#사용할 명사 선정
economic_words_NN = set(economic_NN)
#DTM 적합. df>=10인 단어만을 이용
economic_tf_dtm_ = CountVectorizer(vocabulary=economic_words_NN, min_df=10)
#적용할 문서. stemming이 된 문서를 사용한다.
docu_economic = [' '.join(x) for x in economic_stem]
#문서를 위에서 정의한 DTM fit에 적용
economic_tf_dtm = economic_tf_dtm_.fit_transform(docu_economic)
#사용한 voca를 dict형태로 변환해야 lda를 적용할 수 있다.
economic_voca = dict([(i, s) for i, s in enumerate(economic_tf_dtm_.get_feature_names())]) #tuple을 dict로 변환
#LDA에 적용된 parameter는 topic 100개, alpha=0.1을 사용
economic_LDA = models.ldamodel.LdaModel(corpus=matutils.Sparse2Corpus(economic_tf_dtm), num_topics=100, id2word=economic_voca, passes=20, alpha=0.1)


oshumed_words_NN = set(oshumed_NN)
oshumed_tf_dtm_ = CountVectorizer(vocabulary=oshumed_words_NN, min_df=10, )
docu_oshumed = [' '.join(x) for x in oshumed_stem]
oshumed_tf_dtm = oshumed_tf_dtm_.fit_transform(docu_oshumed)
oshumed_voca = dict([(i, s) for i, s in enumerate(oshumed_tf_dtm_.get_feature_names())]) #tuple을 dict로 변환
oshumed_LDA = models.ldamodel.LdaModel(corpus=matutils.Sparse2Corpus(oshumed_tf_dtm), num_topics=100, id2word=oshumed_voca, passes=20, alpha=0.1)


reuter_words_NN = set(reuter_NN)
reuter_tf_dtm_ = CountVectorizer(vocabulary=reuter_words_NN, min_df=10)
docu_reuter = [' '.join(x) for x in reuter_stem]
reuter_tf_dtm = reuter_tf_dtm_.fit_transform(docu_reuter)
reuter_voca = dict([(i, s) for i, s in enumerate(reuter_tf_dtm_.get_feature_names())]) #tuple을 dict로 변환
reuter_LDA = models.ldamodel.LdaModel(corpus=matutils.Sparse2Corpus(reuter_tf_dtm), num_topics=100, id2word=reuter_voca, passes=20, alpha=0.1)




#per document topic distribution
economic_p_d_t = economic_LDA.inference(matutils.Sparse2Corpus(np.transpose(economic_tf_dtm)))
economic_p_d_t = np.array(economic_p_d_t[0])
scaled_economic_p_d_t = list(map(lambda x: x/sum(x), economic_p_d_t))
pickle.dump(scaled_economic_p_d_t, open('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/representation/economic_LDA_model_100.pickle', 'wb'))

oshumed_p_d_t = oshumed_LDA.inference(matutils.Sparse2Corpus(np.transpose(oshumed_tf_dtm)))
oshumed_p_d_t = np.array(oshumed_p_d_t[0])
scaled_oshumed_p_d_t = list(map(lambda x: x/sum(x), oshumed_p_d_t))
pickle.dump(scaled_oshumed_p_d_t, open('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/representation/oshumed_LDA_model_100.pickle', 'wb'))

reuter_p_d_t = reuter_LDA.inference(matutils.Sparse2Corpus(np.transpose(reuter_tf_dtm)))
reuter_p_d_t = np.array(reuter_p_d_t[0])
scaled_reuter_p_d_t = list(map(lambda x: x/sum(x), reuter_p_d_t))
pickle.dump(scaled_reuter_p_d_t, open('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/representation/reuter_LDA_model_100.pickle', 'wb'))














#####################################################################################################
# Doc2Vec
#####################################################################################################
from collections import namedtuple
from gensim.models import doc2vec

config = {'size':100, 'dm_concat':1, 'dm':1, 'min_count':2} #100차원 공간에 임베딩, dm=1(pv-dm 적용), dm_concat=1(1-layer의 hidden node 구성시 word vector와 paragraph vector concatenate. 논문상에서 average보다 concatenate가 좋다고 얘기함), min_count=2(최소 2회 이상 출현 단어)

#namedtuple 형으로 만들어줘야 doc2vec을 실행할 수 있다.
economic_tagged_document = namedtuple('economic_tagged_document', ['words', 'tags'])
#namedtuple 만드는 과정
economic_tagged_tr_document = [economic_tagged_document(words, [tags]) for tags, words in enumerate(economic_stem)]
#doc2vec 객체 생성 및 하이퍼파라미터 설정
economic_doc_vectorizer = doc2vec.Doc2Vec(**config)
#vocabulary 선정
economic_doc_vectorizer.build_vocab(economic_tagged_tr_document)
#100번 epoch
for epoch in range(100):
    print(epoch)
    #training document로 학습
    economic_doc_vectorizer.train(economic_tagged_tr_document)
    #learning rate decay
    economic_doc_vectorizer.alpha -= 0.002
    #최소
    economic_doc_vectorizer.min_alpha = economic_doc_vectorizer.alpha


oshumed_tagged_document = namedtuple('oshumed_tagged_document', ['words', 'tags'])
oshumed_tagged_tr_document = [oshumed_tagged_document(words, [tags]) for tags, words in enumerate(oshumed_stem)]
oshumed_doc_vectorizer = doc2vec.Doc2Vec(**config)
oshumed_doc_vectorizer.build_vocab(oshumed_tagged_tr_document)
for epoch in range(100):
    print(epoch)
    oshumed_doc_vectorizer.train(oshumed_tagged_tr_document)
    oshumed_doc_vectorizer.alpha -= 0.002
    oshumed_doc_vectorizer.min_alpha = oshumed_doc_vectorizer.alpha
oshumed_doc2vec = np.asarray(oshumed_doc_vectorizer.docvecs)


reuter_tagged_document = namedtuple('reuter_tagged_document', ['words', 'tags'])
reuter_tagged_tr_document = [reuter_tagged_document(words, [tags]) for tags, words in enumerate(reuter_stem)]
reuter_doc_vectorizer = doc2vec.Doc2Vec(**config)
reuter_doc_vectorizer.build_vocab(reuter_tagged_tr_document)
for epoch in range(100):
    print(epoch)
    reuter_doc_vectorizer.train(reuter_tagged_tr_document)
    reuter_doc_vectorizer.alpha -= 0.002
    reuter_doc_vectorizer.min_alpha = reuter_doc_vectorizer.alpha
reuter_doc2vec = np.asarray(reuter_doc_vectorizer.docvecs)


#document embedding
economic_doc2vec = np.asarray(economic_doc_vectorizer.docvecs)
pickle.dump(economic_doc2vec, open('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/representation/economic_Doc2Vec_model_100.pickle', 'wb'))
oshumed_doc2vec = np.asarray(oshumed_doc_vectorizer.docvecs)
pickle.dump(oshumed_doc2vec, open('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/representation/oshumed_Doc2Vec_model_100.pickle', 'wb'))
reuter_doc2vec = np.asarray(reuter_doc_vectorizer.docvecs)
pickle.dump(reuter_doc2vec, open('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/representation/reuter_Doc2Vec_model_100.pickle', 'wb'))