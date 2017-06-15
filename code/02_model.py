import numpy as np
import pandas as pd
import pickle
from collections import Counter
from gensim.models import doc2vec
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics

### dataset
economic = pd.read_csv('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/economic_news_article/economic_news_dataset.csv', delimiter=',', encoding='utf-8')
oshumed = pd.read_csv('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/ohsumed/ohsumed_dataset.csv', delimiter=',', encoding='utf-8')
reuter = pd.read_csv('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/reuter/reuters_dataset.csv', delimiter=',', encoding='utf-8')

### Labels
#economic : 'yes', 'no'만을 사용
economic_error = pickle.load(open('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/representation/economic_error.pickle', 'rb'))
economic_label = np.array(economic.iloc[:,0])
#error난 문서 몇개 제거
economic_label = np.delete(economic_label, economic_error)
#'not sure'라는 label 가진 문서 삭제
notsure = list(np.where(economic_label == 'not sure')[0])
economic_label = np.delete(economic_label, notsure)
#yes를 1로, no를 0으로 정의
eco_classnames, economic_label = np.unique(economic_label, return_inverse=True)

#oshumed
oshumed_error = pickle.load(open('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/representation/oshumed_error.pickle', 'rb'))
oshumed_label = np.array(oshumed.iloc[:,0])
#error난 문서 몇개 제거
oshumed_label = np.delete(oshumed_label, oshumed_error)
#"C04"와 "C14" label만을 사용
osh_C04 = np.where(oshumed_label == 'C04')[0]
osh_C14 = np.where(oshumed_label == 'C14')[0]
uses = np.r_[osh_C04, osh_C14]
oshumed_label = oshumed_label[uses]
#C04를 1로, C14를 0으로 정의
osh_classnames, oshumed_label = np.unique(oshumed_label, return_inverse=True)

#reuter
reuter_label = np.array(reuter.iloc[:,0])
#earn과 earn이 아닌 non-earn으로 2진분류 문제로 정의
reuter_label[np.where(reuter_label!='earn')[0]] = 'non earn'
#non earn을 1로, earn을 0으로 정의
reu_classnames, reuter_label = np.unique(reuter_label, return_inverse=True)


### TF-IDF
economic_tfidf_dtm = pickle.load(open('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/representation/economic_tf_idf_100.pickle', 'rb'))
economic_tfidf_dtm = np.delete(economic_tfidf_dtm, notsure, axis=0)
oshumed_tfidf_dtm = pickle.load(open('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/representation/oshumed_tf_idf_100.pickle', 'rb'))
oshumed_tfidf_dtm = oshumed_tfidf_dtm[uses]
reuter_tfidf_dtm = pickle.load(open('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/representation/reuter_tf_idf_100.pickle', 'rb'))

###LDA
economic_LDA = pickle.load(open('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/representation/economic_LDA_model_100.pickle', 'rb'))
economic_LDA = np.delete(economic_LDA, notsure, axis=0)
oshumed_LDA = pickle.load(open('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/representation/oshumed_LDA_model_100.pickle', 'rb'))
oshumed_LDA = np.array(oshumed_LDA)[uses]
reuter_LDA = pickle.load(open('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/representation/reuter_LDA_model_100.pickle', 'rb'))
reuter_LDA = np.array(reuter_LDA)

### Doc2Vec
economic_doc2vec = pickle.load(open('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/representation/economic_Doc2Vec_model_100.pickle', 'rb'))
economic_doc2vec = np.delete(economic_doc2vec, notsure, axis=0)
oshumed_doc2vec = pickle.load(open('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/representation/oshumed_Doc2Vec_model_100.pickle', 'rb'))
oshumed_doc2vec = oshumed_doc2vec[uses]
reuter_doc2vec = pickle.load(open('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/data/representation/reuter_Doc2Vec_model_100.pickle', 'rb'))









#######################################################################################################################
###Naive Bayesian
def NaiveBayesian(X, y, iter=100):
    X = X
    y = y

    accuracy = []
    recall = []
    precision = []
    f1_measure = []
    for _ in range(iter):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        NB = GaussianNB()
        NB_fit = NB.fit(X=X_train, y=y_train)
        y_pred = NB_fit.predict(X_test)
        if all(y_pred == 0):
            y_pred[-1] = 1
        elif all(y_pred == 1):
            ypred[-1] = 0

        acc = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
        accuracy.append(acc)
        rec = metrics.recall_score(y_true=y_test, y_pred=y_pred)
        recall.append(rec)
        pre = metrics.precision_score(y_true=y_test, y_pred=y_pred)
        precision.append(pre)
        f1 = metrics.f1_score(y_true=y_test, y_pred=y_pred)
        f1_measure.append(f1)

    output = np.c_[accuracy, recall, precision, f1_measure]
    return output

###Decision Tree
def Tree(X, y, iter=100):
    X = X
    y = y

    accuracy = []
    recall = []
    precision = []
    f1_measure = []
    for _ in range(iter):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        Tree = tree.DecisionTreeClassifier()
        Tree_fit = Tree.fit(X=X_train, y=y_train)
        y_pred = Tree_fit.predict(X_test)
        if all(y_pred == 0):
            y_pred[-1] = 1
        elif all(y_pred == 1):
            ypred[-1] = 0

        acc = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
        accuracy.append(acc)
        rec = metrics.recall_score(y_true=y_test, y_pred=y_pred)
        recall.append(rec)
        pre = metrics.precision_score(y_true=y_test, y_pred=y_pred)
        precision.append(pre)
        f1 = metrics.f1_score(y_true=y_test, y_pred=y_pred)
        f1_measure.append(f1)

    output = np.c_[accuracy, recall, precision, f1_measure]
    return output


#economic - NB
economic_tfidf_NB = NaiveBayesian(X=economic_tfidf_dtm, y=economic_label)
economic_LDA_NB = NaiveBayesian(X=economic_LDA, y=economic_label)
economic_doc2vec_NB = NaiveBayesian(X=economic_doc2vec, y=economic_label)
#economic - Tree
economic_tfidf_Tree = Tree(X=economic_tfidf_dtm, y=economic_label)
economic_LDA_Tree = Tree(X=economic_LDA, y=economic_label)
economic_doc2vec_Tree = Tree(X=economic_doc2vec, y=economic_label)


#oshumed - NB
oshumed_tfidf_NB = NaiveBayesian(X=oshumed_tfidf_dtm, y=oshumed_label)
oshumed_LDA_NB = NaiveBayesian(X=oshumed_LDA, y=oshumed_label)
oshumed_doc2vec_NB = NaiveBayesian(X=oshumed_doc2vec, y=oshumed_label)
#oshumed - Tree
oshumed_tfidf_Tree = NaiveBayesian(X=oshumed_tfidf_dtm, y=oshumed_label)
oshumed_LDA_Tree = NaiveBayesian(X=oshumed_LDA, y=oshumed_label)
oshumed_doc2vec_Tree = NaiveBayesian(X=oshumed_doc2vec, y=oshumed_label)


#reuter - NB
reuter_tfidf_NB = NaiveBayesian(X=reuter_tfidf_dtm, y=reuter_label)
reuter_LDA_NB = NaiveBayesian(X=reuter_LDA, y=reuter_label)
reuter_doc2vec_NB = NaiveBayesian(X=reuter_doc2vec, y=reuter_label)
#reuter - Tree
reuter_tfidf_Tree = NaiveBayesian(X=reuter_tfidf_dtm, y=reuter_label)
reuter_LDA_Tree = NaiveBayesian(X=reuter_LDA, y=reuter_label)
reuter_doc2vec_Tree = NaiveBayesian(X=reuter_doc2vec, y=reuter_label)



######################################################################################################################
###Ensemble
def Ensemble(X1, X2, X3, y, model, iter=100, major='voting'):
    X1 = X1
    X2 = X2
    X3 = X3
    y = y

    accuracy = []
    recall = []
    precision = []
    f1_measure = []
    for _ in range(iter):
        tr_ind = np.random.choice(a=np.arange(len(X1)), size=int(np.round(len(X1)*0.7)), replace=False)
        X1_train, X1_test = X1[tr_ind], np.delete(X1, tr_ind, axis=0)
        X2_train, X2_test = X2[tr_ind], np.delete(X2, tr_ind, axis=0)
        X3_train, X3_test = X3[tr_ind], np.delete(X3, tr_ind, axis=0)
        y_train, y_test = y[tr_ind], np.delete(y, tr_ind)

        if model=='NB':
            model = GaussianNB()
        else:
            model = tree.DecisionTreeClassifier()

        model1 = model.fit(X=X1_train, y=y_train)
        model2 = model.fit(X=X2_train, y=y_train)
        model3 = model.fit(X=X3_train, y=y_train)

        if major == 'voting':
            pred1 = model1.predict(X1_test)
            pred2 = model2.predict(X2_test)
            pred3 = model3.predict(X3_test)

            PRED = np.c_[pred1, pred2, pred3]
            mean_PRED = np.mean(PRED, axis=1) > 0.5
            y_pred = mean_PRED.astype(int)

            if all(y_pred==0):
                y_pred[-1] = 1
            elif all(y_pred==1):
                ypred[-1] = 0

        else:
            pred1 = model1.predict_proba(X1_test)
            pred2 = model2.predict_proba(X2_test)
            pred3 = model3.predict_proba(X3_test)

            PRED = np.stack([pred1, pred2, pred3], axis=2)
            mean_PRED = np.mean(PRED, axis=2)
            y_pred = np.argmax(mean_PRED, axis=1)

            if all(y_pred == 0):
                y_pred[-1] = 1
            elif all(y_pred == 1):
                ypred[-1] = 0

        acc = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
        accuracy.append(acc)
        rec = metrics.recall_score(y_true=y_test, y_pred=y_pred)
        recall.append(rec)
        pre = metrics.precision_score(y_true=y_test, y_pred=y_pred)
        precision.append(pre)
        f1 = metrics.f1_score(y_true=y_test, y_pred=y_pred)
        f1_measure.append(f1)

    output = np.c_[accuracy, recall, precision, f1_measure]
    return output


#NB - Ensemble
economic_NB_Ensemble_voting = Ensemble(X1=economic_tfidf_dtm, X2=economic_LDA, X3=economic_doc2vec, y=economic_label, model='NB', major='voting')
economic_NB_Ensemble_prob = Ensemble(X1=economic_tfidf_dtm, X2=economic_LDA, X3=economic_doc2vec, y=economic_label, model='NB', major='prob')

oshumed_NB_Ensemble_voting = Ensemble(X1=oshumed_tfidf_dtm, X2=oshumed_LDA, X3=oshumed_doc2vec, y=oshumed_label, model='NB', major='voting')
oshumed_NB_Ensemble_prob = Ensemble(X1=oshumed_tfidf_dtm, X2=oshumed_LDA, X3=oshumed_doc2vec, y=oshumed_label, model='NB', major='prob')

reuter_NB_Ensemble_voting = Ensemble(X1=reuter_tfidf_dtm, X2=reuter_LDA, X3=reuter_doc2vec, y=reuter_label, model='NB', major='voting')
reuter_NB_Ensemble_prob = Ensemble(X1=reuter_tfidf_dtm, X2=reuter_LDA, X3=reuter_doc2vec, y=reuter_label, model='NB', major='prob')

#Tree - Ensemble
economic_Tree_Ensemble_voting = Ensemble(X1=economic_tfidf_dtm, X2=economic_LDA, X3=economic_doc2vec, y=economic_label, model='Tree', major='voting')
economic_Tree_Ensemble_prob = Ensemble(X1=economic_tfidf_dtm, X2=economic_LDA, X3=economic_doc2vec, y=economic_label, model='Tree', major='prob')

oshumed_Tree_Ensemble_voting = Ensemble(X1=oshumed_tfidf_dtm, X2=oshumed_LDA, X3=oshumed_doc2vec, y=oshumed_label, model='Tree', major='voting')
oshumed_Tree_Ensemble_prob = Ensemble(X1=oshumed_tfidf_dtm, X2=oshumed_LDA, X3=oshumed_doc2vec, y=oshumed_label, model='Tree', major='prob')

reuter_Tree_Ensemble_voting = Ensemble(X1=reuter_tfidf_dtm, X2=reuter_LDA, X3=reuter_doc2vec, y=reuter_label, model='Tree', major='voting')
reuter_Tree_Ensemble_prob = Ensemble(X1=reuter_tfidf_dtm, X2=reuter_LDA, X3=reuter_doc2vec, y=reuter_label, model='Tree', major='prob')



x=economic_tfidf_NB
def mean_sd(x):
    mean = np.round(np.mean(x, axis=0), decimals=4)
    sd = np.round(np.std(x, axis=0), decimals=4)
    a = str(mean[0]) + '(' + str(sd[0]) + ')'
    b = str(mean[1]) + '(' + str(sd[1]) + ')'
    c = str(mean[2]) + '(' + str(sd[2]) + ')'
    d = str(mean[3]) + '(' + str(sd[3]) + ')'
    out = [a, b, c, d]
    return out

outputJAM = np.c_[mean_sd(economic_tfidf_NB),
mean_sd(economic_LDA_NB),
mean_sd(economic_doc2vec_NB),
mean_sd(economic_NB_Ensemble_voting),
mean_sd(economic_NB_Ensemble_prob),
mean_sd(economic_tfidf_Tree),
mean_sd(economic_LDA_Tree),
mean_sd(economic_doc2vec_Tree),
mean_sd(economic_Tree_Ensemble_voting),
mean_sd(economic_Tree_Ensemble_prob),

mean_sd(oshumed_tfidf_NB),
mean_sd(oshumed_LDA_NB),
mean_sd(oshumed_doc2vec_NB),
mean_sd(oshumed_NB_Ensemble_voting),
mean_sd(oshumed_NB_Ensemble_prob),
mean_sd(oshumed_tfidf_Tree),
mean_sd(oshumed_LDA_Tree),
mean_sd(oshumed_doc2vec_Tree),
mean_sd(oshumed_Tree_Ensemble_voting),
mean_sd(oshumed_Tree_Ensemble_prob),

mean_sd(reuter_tfidf_NB),
mean_sd(reuter_LDA_NB),
mean_sd(reuter_doc2vec_NB),
mean_sd(reuter_NB_Ensemble_voting),
mean_sd(reuter_NB_Ensemble_prob),
mean_sd(reuter_tfidf_Tree),
mean_sd(reuter_LDA_Tree),
mean_sd(reuter_doc2vec_Tree),
mean_sd(reuter_Tree_Ensemble_voting),
mean_sd(reuter_Tree_Ensemble_prob)]

outputJAM = outputJAM.T
outputJAM1 = pd.DataFrame(outputJAM)
outputJAM1.to_csv('C:/Users/deokseong/Desktop/Lab/2017-1/강의/Pattern recognition/과제/team project/OUTPUT/PRML_performance.csv')



