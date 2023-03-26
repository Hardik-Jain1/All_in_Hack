from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    pm25 = request.form['pm25']
    pm10 = request.form['pm10']
    no = request.form['no']
    no2 = request.form['no2']
    nox = request.form['nox']
    nh3 = request.form['nh3']
    co = request.form['co']
    so2 = request.form['so2']
    o3 = request.form['o3']
    benzene = request.form['benzene']
    toluene = request.form['toluene']
    xylene = request.form['xylene']
    
    # Convert form data to feature vector
    features = [[pm25, pm10, no, no2, nox, nh3, co, so2, o3, benzene, toluene, xylene]]
    
    # Make prediction
    prediction = model.predict(features)
    
    # Render the result template with prediction
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)