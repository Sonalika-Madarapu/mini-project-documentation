from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('model.pkl')



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data and convert it to a float array
            features = [float(x) for x in request.form.values()]
            input_data = np.array([features])
            
            # Ensure model is of correct type before prediction
            if hasattr(model, 'predict'):
                prediction = model.predict(input_data)
                output = prediction[0]
                result_text = "Reservation Cancelled" if output == 1 else "Reservation Not Cancelled"
                return render_template('result.html', prediction_text=f'Prediction: {result_text}')
            else:
                return "Loaded object is not a model"
        except Exception as e:
            return f"An error occurred: {e}"
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
