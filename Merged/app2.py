import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import joblib


svm_classifier_modle_for_diabetes = pickle.load(open("svm_classifier_modle_for_diabetes.pkl", "rb"))
data_max = pickle.load(open("data_max.pkl", "rb"))
data_min = pickle.load(open("data_min.pkl", "rb"))
data_mean = pickle.load(open("data_mean.pkl", "rb"))
data_k = pickle.load(open("kidney.pkl", "rb"))

def normalize_input_values(input_initial_value, attribute):
    input_normalized_value = (input_initial_value - data_mean[attribute]) / (data_max[attribute] - data_min[attribute])
    return input_normalized_value

app = Flask(__name__)
model_heart = pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/heart')
def heart():
    return render_template('heart.html', prediction_text='')

@app.route('/heart_predict', methods=["POST"])
def heart_predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = [ "age", "sex","cp","trestbps", "chol", "fbs",
                       "restecg", "thalach", "exang", "oldpeak", "slope","ca",
                        "thal"]
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model_heart.predict(df)
        
    if output == 1:
        res_val = "** heart disease **"
    else:
        res_val = "no heart disease "
        

    return render_template('heart.html', prediction_text=f'Patient has {res_val}')

@app.route('/daibetes', methods=["GET"])
def daibetes():
    return render_template("daibetes.html", data="")

@app.route('/daibetes_predict', methods=["POST"])
def daibetes_predict():
    pregnancies                 = float(request.form[    "Pregnancies"                 ])
    glucose                     = float(request.form[    "Glucose"                     ])
    blood_pressure              = float(request.form[    "BloodPressure"               ])
    skin_thickness              = float(request.form[    "SkinThickness"               ])
    insulin                     = float(request.form[    "Insulin"                     ])
    bmi                         = float(request.form[    "BMI"                         ])
    diabetes_pedigree_function  = float(request.form[    "DiabetesPedigreeFunction"    ])
    age                         = float(request.form[    "Age"                         ])

    
    pregnancies                 = normalize_input_values(pregnancies, 'Pregnancies')
    glucose                     = normalize_input_values(glucose, 'Glucose')
    blood_pressure              = normalize_input_values(blood_pressure, 'BloodPressure')
    skin_thickness              = normalize_input_values(skin_thickness, 'SkinThickness')
    insulin                     = normalize_input_values(insulin, 'Insulin')
    bmi                         = normalize_input_values(bmi, 'BMI')
    diabetes_pedigree_function  = normalize_input_values(diabetes_pedigree_function, 'DiabetesPedigreeFunction')
    age                         = normalize_input_values(age, 'Age')

    
    to_predict = [
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree_function,
        age
    ]

    
    to_predict_as_numpy_array           = np.asarray(to_predict)
    to_predict_as_numpy_array_reshaped  = to_predict_as_numpy_array.reshape(1, -1)
    prediction_of_to_predict            = svm_classifier_modle_for_diabetes.predict(to_predict_as_numpy_array_reshaped)

    if prediction_of_to_predict[0] == 1:
        result = "The patient has diabetes."
    elif prediction_of_to_predict[0] == 0:
        result = "The patient doesn't have diabetes."
    return render_template("daibetes.html", data=result)

@app.route('/catract', methods=["GET"])
def catract():
    return render_template("catract.html", data="")
"""
@app.route('/catract_predict', methods=["POST"])
def catract_predict():
    eye = str(request.form['imgeye'])
    # print(eye)
    result = 0
    if int(eye[0])%2==0:
        result = 'Catract Detected'
    else:
        result = 'No Catract Detected'
    return render_template("catract.html", data=result)"""

@app.route('/kidney', methods=['GET'])
def kidney():
    return render_template("kidney.html", data="")

@app.route('/diet')
def diet():
    return render_template("diet.html")



if __name__ == "__main__":
    app.run()


