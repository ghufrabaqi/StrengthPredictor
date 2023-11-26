import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request,app,url_for,jsonify


app = Flask(__name__) # defining flask app

model = pickle.load(open('CCS_model_randomforest.pkl','rb'))  # loading the randomforest model


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    output= model.predict(data)
    print(output[0])
    return jsonify(output[0])

""" 
    if request.method == "POST":
 # ['Cement', 'Blast_furnace_slag', 'Fly_ash', 'Water','Superplasticizer', 'Coarse_aggregate', 'Fine_aggregate', 'Age']
        list = [request.form.get('Cement'), 
                request.form.get('Blast_furnace_slag'),
                request.form.get('Fly_ash'),
                request.form.get('Water'),
                request.form.get('Superplasticizer'), 
                request.form.get('Coarse_aggregate'),
                request.form.get('Fine_aggregate'),
                request.form.get('Age')]  # list of inputs



        final_features = np.array(list).reshape(1,8)
        df = pd.DataFrame(final_features)

        output = model.predict(df)
        result = "%.2f" % round(output[0])

        
        return render_template('index.html',
                               prediction_text=f"Concrete compressive strength is {result} MPa")



if __name__ == "__main__":
    app.run(debug=True)