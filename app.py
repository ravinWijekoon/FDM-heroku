import flask
import pickle
import pandas as pd
import urllib.request
import json
import ast
from flask import url_for

app = flask.Flask(__name__, template_folder='templates')

with open('framingham_classifier_Logistic_regression_new.pkl', 'rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('main.html')
    if flask.request.method == 'POST':
        age = flask.request.form['age']
        sysBP = flask.request.form['sysBP']
        diaBP = flask.request.form['diaBP']
        glucose = flask.request.form['glucose']
        #diabetes = flask.request.form['diabetes']
        male = flask.request.form['male']
        BPMeds = flask.request.form['BPMeds']
        totChol = flask.request.form['totChol']
        BMI = flask.request.form['BMI']
        prevalentStroke = flask.request.form['prevalentStroke']
        prevalentHyp = flask.request.form['prevalentHyp']
        pregnantNo = flask.request.form['pregnantNo']
        plasmaGlucoseConc = flask.request.form['plasmaGlucoseConc']
        tricepsThickness = flask.request.form['tricepsThickness']
        SerumInsulin = flask.request.form['SerumInsulin']
        diabPedigreeFunc = flask.request.form['diabPedigreeFunc']

        data1 = {
            "Inputs": {
                "input1":
                    [
                        {
                            'Number of times pregnant': pregnantNo,
                            'Plasma glucose concentration a 2 hours in an oral glucose tolerance test': plasmaGlucoseConc,
                            'Diastolic blood pressure (mm Hg)': diaBP,
                            'Triceps skin fold thickness (mm)': tricepsThickness,
                            '2-Hour serum insulin (mu U/ml)': SerumInsulin,
                            'Body mass index (weight in kg/(height in m)^2)': BMI,
                            'Diabetes pedigree function': diabPedigreeFunc,
                            'Age (years)': age,
                            'Class variable (0 or 1)': "0",
                        }
                    ],
            },
            "GlobalParameters": {}
        }
        body = str.encode(json.dumps(data1))

        url = 'https://ussouthcentral.services.azureml.net/workspaces/13c077d4051e4e1088654297b2bbcb04/services/934466005a2243948e5d6b46d9cdec64/execute?api-version=2.0&format=swagger'
        api_key = 'u4bfO9QM3gPLQ4nbSXiFNXP/h4B3yO0QE1lQy0/GOSqPwgOTFwAyWr4WXEYKj4tfrvZ/mIvRZpH2b5bn9QxHgg=='  # Replace this with the API key for the web service
        headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}

        req = urllib.request.Request(url, body, headers)

        try:
            response = urllib.request.urlopen(req)
            result = response.read()
            my_json = result.decode('utf8').replace("'", '"')
            data = json.loads(my_json)
            s = json.dumps(data, indent=4, sort_keys=True)
            FinalData = data["Results"]['output1']
            res = str(FinalData)[1:-1]
            json_data = ast.literal_eval(res)
            FinalOutputAzure = json_data["Scored Labels"]
            NewDiabetesColumn = json_data["Scored Labels"]

        except urllib.error.HTTPError as error:
            print("The request failed with status code: " + str(error.code))
            # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
            print(error.info())
            print(json.loads(error.read().decode("utf8", 'ignore')))

        input_variables = pd.DataFrame(
            [[age, sysBP, diaBP, glucose, NewDiabetesColumn, male, BPMeds, totChol, BMI, prevalentStroke, prevalentHyp]],
            columns=['age', 'sysBP', 'diaBP', 'glucose', 'diabetes', 'male', 'BPMeds', 'totChol', 'BMI',
                     'prevalentStroke', 'prevalentHyp'],
            dtype=float)

        prediction = model.predict(input_variables)[0]
        return flask.render_template('main.html',
                                     original_input={'Age': age,
                                                     'Systolic BP': sysBP,
                                                     'Diastolic BP': diaBP,
                                                     'Glucose': glucose,
                                                     'Diabetes': NewDiabetesColumn,
                                                     'Gender': male,
                                                     'BP Medication': BPMeds,
                                                     'Total Cholesterol': totChol,
                                                     'BMI': BMI,
                                                     'Prevalent Stroke': prevalentStroke,
                                                     'Number of times pregnant': pregnantNo,
                                                     'Plasma glucose concentration a 2 hours in an oral glucose tolerance test': plasmaGlucoseConc,
                                                     '2-Hour serum insulin (mu U/ml)': SerumInsulin,
                                                     'Triceps skin fold thickness (mm)': tricepsThickness,
                                                     'Diabetes pedigree function': diabPedigreeFunc,
                                                     'Prevalent Hypertension': prevalentHyp},
                                     result=prediction,
                                     azureresult=FinalOutputAzure,
                                     )


if __name__ == '__main__':
    app.run()
