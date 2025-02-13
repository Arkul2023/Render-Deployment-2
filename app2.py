from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

# Load the trained model
with open("grid_search3.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__, template_folder=os.path.abspath("templates2"))

@app.route('/')
def home():
    return render_template('index2.html', prediction_text = "")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form
        int_features = [float(x) for x in request.form.values()]
        final_features = np.array(int_features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(final_features)[0]

        output = 'Continue using the service' if prediction == 1 else 'Stop using the service'
    
        return render_template('index2.html', prediction_text=f'Customer is likely to {output}')

    except Exception as e:
        return render_template('index2.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
