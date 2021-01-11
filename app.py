# importing the necessary dependencies
from flask import Flask, render_template, request, jsonify
import sklearn
import pickle
import joblib

app = Flask(__name__)  # initializing a flask app


@app.route('/')  # route to display the home page
def homePage():
    return render_template("index.html")





@app.route('/predict', methods=['POST'])  # route to show the predictions in a web UI
def index():
    # # reading the inputs given by the user
    Pregnancies = (request.form['Pregnancies'])
    Glucose = (request.form['Glucose'])
    BloodPressure = (request.form['BloodPressure'])
    SkinThickness = (request.form['SkinThickness'])
    Insulin = (request.form['Insulin'])
    BMI = (request.form['BMI'])
    DiabetesPedigree = (request.form['DiabetesPedigree'])
    Age = (request.form['Age'])


    filename = "stacking_diabetes_suraj.sav"
    loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage
    # predictions using the loaded model file
    prediction = loaded_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness,Insulin, BMI, DiabetesPedigree, Age]])
    print('prediction is', prediction)
    # showing the prediction results in a UI
    return render_template('results.html', prediction=prediction[0])


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
