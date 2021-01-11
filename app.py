# importing the necessary dependencies
from flask import Flask, render_template, request, jsonify
import sklearn
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV
import pickle
import joblib

app = Flask(__name__)  # initializing a flask app


@app.route('/')  # route to display the home page
def homePage():
    return render_template("index.html")





@app.route('/predict', methods=['POST'])  # route to show the predictions in a web UI
def index():
    Pregnancies = (request.form['Pregnancies'])
    Glucose = (request.form['Glucose'])
    BloodPressure = (request.form['BloodPressure'])
    SkinThickness = (request.form['SkinThickness'])
    Insulin = (request.form['Insulin'])
    BMI = (request.form['BMI'])
    DiabetesPedigree = (request.form['DiabetesPedigree'])
    Age = (request.form['Age'])
    data = pd.read_csv("diabetes.csv")



    X = data.drop(columns='Outcome')
    y = data['Outcome']

    # let's divide our dataset into training set and hold out set by 50%
    train, val_train, test, val_test = train_test_split(X, y, test_size=0.5, random_state=355)

    # let's split the training set again into training and test dataset
    x_train, x_test, y_train, y_test = train_test_split(train, test, test_size=0.2, random_state=355)

    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)

    svm = SVC()
    svm.fit(x_train, y_train)

    predict_val1 = knn.predict(val_train)
    predict_val2 = svm.predict(val_train)

    predict_val = np.column_stack((predict_val1, predict_val2))

    predict_test1 = knn.predict(x_test)
    predict_test2 = svm.predict(x_test)

    predict_test = np.column_stack((predict_test1, predict_test2))

    rand_clf = RandomForestClassifier()



    rand_clf = RandomForestClassifier(criterion='gini', max_features='auto', min_samples_leaf=1, min_samples_split=4, n_estimators=90)

    rand_clf.fit(predict_val, val_test)

    a = knn.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigree,Age]])
    b = svm.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigree,Age]])
    c = np.column_stack((a, b))

    d = rand_clf.predict(c)

    # # reading the inputs given by the user




    return render_template('results.html', prediction=d)


@app.route('/poo', methods=['POST', 'GET'])  # route to show the predictions in a web UI
def postm():

    print(request.get_json(force=True))
    data= request.get_json(force=True)
    Pregnancies = (data['Pregnancies'])
    Glucose = (data['Glucose'])
    BloodPressure = (data['BloodPressure'])
    SkinThickness = (data['SkinThickness'])
    Insulin = (data['Insulin'])
    BMI = (data['BMI'])
    DiabetesPedigree = (data['DiabetesPedigree'])
    Age = (data['Age'])

    filename = "stacking_diabetes_suraj.sav"
    loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage
    # predictions using the loaded model file
    prediction = loaded_model.predict(
        [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigree, Age]])
    print('prediction is', prediction)
    # showing the prediction results in a UI
    return jsonify({'Prediction': prediction})


if __name__ == "__main__":
    app.run(debug=True)  # running the app
