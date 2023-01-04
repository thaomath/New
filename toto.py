# Import all packages and libraries
import pandas as pd
import numpy as np
from flask import Flask, render_template, url_for, request, jsonify
import pickle
import math
import base64

from lightgbm import LGBMClassifier

data = pd.read_csv('C:\\Users\\LEMuon\\Documents\\Thao_Openclassrooms\\P7_final\\New\\Data\\P7_data_test_20features_importance_std_sample.csv', sep=",")

# Loading model to compare the results
model = pickle.load(open('C:\\Users\\LEMuon\\Documents\\Thao_Openclassrooms\\P7_final\\New\\Data\\model_complete.pkl','rb'))

seuil = 0.52

#data = pd.read_csv('data_predict_api.csv', sep=",")
#list_id_client = list(data['SK_ID_CURR'].unique())
list_id_client = data['SK_ID_CURR'].tolist()

#model = pickle.load(open('trained_gbc_model.pkl', 'rb'))

app= Flask(__name__)


@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/predict/all', methods=['GET'])
def api_all():
    print("Step 2")
    return jsonify(list_id_client)


@app.route('/predict', methods = ['GET'])
def predict():
    #id = request.form['id_client']
    if 'id' in request.args:
        id = int(request.args['id'])


    #id = int(id)
    if id not in list_id_client:
        prediction="Ce client n'est pas répertorié"
    else :
        #prediction="Ce client est répertorié"
  
        X = data[data['SK_ID_CURR'] == id]
        X = X.drop(['SK_ID_CURR'], axis=1)

        probability_default_payment = model.predict_proba(X)[:, 1]
        if probability_default_payment >= seuil:
            prediction = "Client défaillant: prêt refusé"
        else:
            prediction = "Client non défaillant:  prêt accordé"
    return render_template('dashboard.html',  prediction_text=prediction)
   

    

# Define endpoint for flask
#app.add_url_rule('/predict', 'predict', predict)

# Run app.
if __name__ == '__main__':
    app.run()