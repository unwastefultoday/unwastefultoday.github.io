from flask import Flask
from flask import render_template, request
import joblib
import numpy as np

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.route('/')
def index():
    return render_template('/index.html')


@app.route('/hello_page')
def myfunction():
    return render_template('/hello.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        form_data_list = list(form_data.values())
        form_data_list = list(map(float, form_data_list))
        form_data_list = np.array(form_data_list).reshape(1, 10)
        model = joblib.load('bodyfat_linearreg.sav')
        prediction = model.predict(form_data_list)
        return render_template("/result.html", fatpercent = prediction)


@app.route('/index.html')
def msg():
    return 'Hello World'


if (__name__ == '__main__'):
    app.run(debug=True)
