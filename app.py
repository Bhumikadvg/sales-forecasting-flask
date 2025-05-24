from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_val = float(request.form['ad_spend'])
        prediction = model.predict(np.array([[input_val]]))[0]
        return render_template('result.html', prediction=round(prediction, 2), input_val=input_val)
    except ValueError:
        return abort(400, description="Invalid input. Please enter a valid number.")
    except Exception as e:
        return abort(500, description=f"An error occurred: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
