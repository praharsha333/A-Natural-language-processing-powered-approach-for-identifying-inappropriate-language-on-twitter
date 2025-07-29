from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
# from .models import predictions
import os
import pandas as pd


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

def DatasetView(request):
    path = os.path.join(settings.MEDIA_ROOT, 'labeled_data.csv' )
    df = pd.read_csv(path,  encoding = "ISO-8859-1")
    df = df[['class','tweet']]
    # print(df.head())
    df = df.to_html(index=None)
    return render(request, 'users/viewdataset.html', {'data': df})  

def user_ml_code(request):
    from .algorithms import  textpreprocessing
    cr_lg = textpreprocessing.start_adboost()   
    cr_rnn = textpreprocessing.start_multi_layer_perceptron()
    cr_rf = textpreprocessing.RandomForest()
    cr_nb = textpreprocessing.GaussianNaiveBayes()
    cr_lr =textpreprocessing.LogisticRegressionModel()
    cr_dt = textpreprocessing.DecisionTree()
    cr_gb =textpreprocessing.GradientBoosting()
    cr_svm = textpreprocessing.start_svm()
    return render(request, 'users/ml_results.html', {'cr_lg': cr_lg,  'cr_svm': cr_svm,'cr_rnn':cr_rnn, 'cr_rf':cr_rf,
                                                    'cr_nb':cr_nb , 'cr_dt':cr_dt,'cr_lr':cr_lr,'cr_gb':cr_gb})


def predict(request):
    if request.method=='POST':
        tweets = request.POST.get('tweets')
        print('-'*100)
        print(tweets)
        import os
        import pandas as pd
        from django.conf import settings
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        import pickle
        from sklearn.model_selection import train_test_split
        path1 = os.path.join(settings.MEDIA_ROOT, 'labeled_data.csv')
        # path = os.path.join(settings.MEDIA_ROOT, 'fakenews.alex')
        df = pd.read_csv(path1, encoding="ISO-8859-1")
        

        # df['hate_speech'] = df.hate_speech.replace({'offensive': 1, 'non-offensive': 0})
                
        print(df['class'])
        X_train, X_test, y_train, y_test = train_test_split(df['tweet'],
                                                            df['class'],test_size=0.2,
                                                            random_state=42)
        # Instantiate the CountVectorizer method
        count_vector = CountVectorizer(stop_words='english', lowercase=True)
        training_data = count_vector.fit_transform(X_train)
        # Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
        testing_data = count_vector.transform(X_test)
        test = count_vector.transform([tweets])
        print('test')
        # model = pickle.load(open(path1, 'rb'))
        model = LogisticRegression()
        model.fit(training_data,y_train)
        pred = model.predict(test)
        if pred[0] == 1:
            msg = 'offensive'
        elif pred[0] == 2:
            msg = 'non-offensive'
        else:
            msg='neither'
        print("===>", pred)
        return render(request, 'users/testform.html', {'tweet': tweets, 'msg': msg})
    else:
        return render(request, 'users/testform.html', {})  