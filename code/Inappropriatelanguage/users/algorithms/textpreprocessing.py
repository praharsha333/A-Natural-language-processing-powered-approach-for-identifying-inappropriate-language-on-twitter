import os

# import matplotlib.pyplot as plt
import pandas as pd
from django.conf import settings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

path1 = os.path.join(settings.MEDIA_ROOT, 'labeled_data.csv')
df_offensive = pd.read_csv(path1, nrows=500)
# plt.figure(figsize=(7, 7))
df_offensive.drop(['Unnamed: 0','count','hate_speech','offensive_language','neither'],axis=1,inplace=True)
# df_offensive[df_offensive['class']==0]['class']=1

# df_offensive["class"].replace({0: 1}, inplace=True)
# df_offensive["class"].replace({2: 0}, inplace=True)
import re
import nltk
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))
def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
df_offensive['tweet']=df_offensive['tweet'].apply(clean_text)
x=df_offensive['tweet']
y=df_offensive['class']
print(df_offensive['class'].unique())
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)


#print(len(x_train), len(y_train))
#print(len(x_test), len(y_test))
type(x_train)
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
x_train_vectorizer=count.fit_transform(x_train)
x_test_vectorizer=count.transform(x_test)
x_train_vectorizer.toarray()
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()

x_train_tfidf = tfidf.fit_transform(x_train_vectorizer)

x_train_tfidf.toarray()
x_test_tfidf = tfidf.transform(x_test_vectorizer)
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix





def start_adboost():
    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier()
    model_vectorizer= model.fit(x_train_tfidf, y_train)
    y_pred=model_vectorizer.predict(x_test_vectorizer)
    cr_lg = classification_report(y_pred, y_test, output_dict=True)
    # cr_k = confusion_matrix(y_pred, y_test)
    # print(cr_k)
    
    
    cr_lg = pd.DataFrame(cr_lg).transpose()
    cr_lg = pd.DataFrame(cr_lg)
    cr_lg = cr_lg.to_html
    return cr_lg


def start_svm():
    from sklearn.svm import SVC
    model = SVC()
    model_vectorizer= model.fit(x_train_tfidf, y_train)
    y_pred=model_vectorizer.predict(x_test_vectorizer)
    cr_svm = classification_report(y_pred, y_test, output_dict=True)
    cr_svm = pd.DataFrame(cr_svm).transpose()
    cr_svm = pd.DataFrame(cr_svm)
    cr_k = confusion_matrix(y_pred, y_test)
    print(cr_k)
    cr_svm = cr_svm.to_html
    return cr_svm

def start_multi_layer_perceptron():
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(10, 10, 10, 10), verbose=True)
    model.fit(x_train_tfidf, y_train)
    y_pred=model.predict(x_test_vectorizer)
    cr_rnn = classification_report(y_pred, y_test, output_dict=True)
    cr_rnn = pd.DataFrame(cr_rnn).transpose()
    cr_rnn = pd.DataFrame(cr_rnn)
    cr_rnn = cr_rnn.to_html
    return cr_rnn

def RandomForest():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    model = RandomForestClassifier(n_estimators=200)
    model_vectorizer= model.fit(x_train_tfidf, y_train)
    y_pred=model_vectorizer.predict(x_test_vectorizer)
    cr_rf = classification_report(y_pred, y_test, output_dict=True)
    cr_rf = pd.DataFrame(cr_rf).transpose()
    cr_rf = pd.DataFrame(cr_rf)
    cr_rf = cr_rf.to_html
    return cr_rf

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

def GaussianNaiveBayes():
    model = GaussianNB()
    model.fit(x_train_tfidf.toarray(), y_train)  # Convert x_train_tfidf to dense array using toarray()
    y_pred = model.predict(x_test_vectorizer.toarray())  # Convert x_test_vectorizer to dense array using toarray()
    accuracy = accuracy_score(y_test, y_pred)
    cr_nb = classification_report(y_test, y_pred, output_dict=True)
    cr_nb = pd.DataFrame(cr_nb).transpose()
    cr_nb = pd.DataFrame(cr_nb)
    cr_nb = cr_nb.to_html
    return cr_nb


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def DecisionTree():
    model = DecisionTreeClassifier()
    model.fit(x_train_tfidf.toarray(), y_train)  # Convert x_train_tfidf to dense array using toarray()
    y_pred = model.predict(x_test_vectorizer.toarray())  # Convert x_test_vectorizer to dense array using toarray()
    accuracy = accuracy_score(y_test, y_pred) * 100  # Multiply by 100 to get the accuracy in percentage
    cr_dt = classification_report(y_test, y_pred, output_dict=True)
    cr_dt = pd.DataFrame(cr_dt).transpose()
    cr_dt = pd.DataFrame(cr_dt)
    cr_dt = cr_dt.to_html
    return cr_dt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def LogisticRegressionModel():
    model = LogisticRegression()
    model.fit(x_train_tfidf.toarray(), y_train)  # Convert x_train_tfidf to dense array using toarray()
    y_pred = model.predict(x_test_vectorizer.toarray())  # Convert x_test_vectorizer to dense array using toarray()
    accuracy = accuracy_score(y_test, y_pred) * 100  # Multiply by 100 to get the accuracy in percentage
    cr_lr = classification_report(y_test, y_pred, output_dict=True)
    cr_lr = pd.DataFrame(cr_lr).transpose()
    cr_lr = pd.DataFrame(cr_lr)
    cr_lr = cr_lr.to_html
    return cr_lr

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

def GradientBoosting():
    model = GradientBoostingClassifier()
    model.fit(x_train_tfidf.toarray(), y_train)  # Convert x_train_tfidf to dense array using toarray()
    y_pred = model.predict(x_test_vectorizer.toarray())  # Convert x_test_vectorizer to dense array using toarray()
    accuracy = accuracy_score(y_test, y_pred) * 100  # Multiply by 100 to get the accuracy in percentage
    cr_gb = classification_report(y_test, y_pred, output_dict=True)
    cr_gb = pd.DataFrame(cr_gb).transpose()
    cr_gb = pd.DataFrame(cr_gb)
    cr_gb = cr_gb.to_html
    return cr_gb