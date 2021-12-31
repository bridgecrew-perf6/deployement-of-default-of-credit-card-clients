import pickle

import numpy as np

from flask import render_template, redirect, url_for,request
from flask import Flask
app = Flask(__name__)

logreg=pickle.load(open('/Users/admin/PycharmProjects/projetML/model.pkl','rb'))


@app.route("/", methods=['GET','POST'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['PAY_1'])
        glucose = int(request.form['PAY_2'])
        bp = int(request.form['PAY_3'])
        st = int(request.form['PAY_4'])
        insulin = int(request.form['PAY_5'])
        bmi = float(request.form['PAY_6'])


        data = np.array([[preg, glucose, bp, st, insulin, bmi]])



        return render_template('result.html', prediction=logreg.predict(data))


if __name__== "__main__":
    app.run(host='localhost', port=5000)