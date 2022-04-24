from flask import Flask, render_template, request
import numpy as np

import pickle

model = pickle.load(open('model.pkl','rb'))
app = Flask('__name__')

@app.route("/")
def index():
    
    return render_template('index.html')

@app.route("/predict",methods = ['POST'])
def placement():
    '''Prediction logic'''
    cgpa = float(request.form.get('cgpa_index'))
    iq = int(request.form.get('iq_index'))
    profile_score = int(request.form.get('profile_score_index'))
    print(cgpa,iq,profile_score)

    result = model.predict(np.array([cgpa,iq,profile_score]).reshape(1,3))
    print(result)

    if result[0]==1:
        final_prediction = "Student is Placed"
    else:
        final_prediction = "Student is not placed"


    return render_template("index.html", Predict = final_prediction)


    


if __name__ == "__main__":
    app.run(debug=True)
