from flask import Flask, render_template, request
import joblib  
import numpy as np

app = Flask(__name__, template_folder='template') 


#loading my pretrained model
model = joblib.load('./air_quality_model.joblib')



@app.route('/')
def form():
    return render_template('front.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        pm25 = float(request.form['pm25'])
        pm10 = float(request.form['pm10'])
        no2 = float(request.form['no2'])
        so2 = float(request.form['so2'])
        co = float(request.form['co'])
        proximity_to_industrial_areas = float(request.form['proximity_to_industrial_areas'])
        population_density = float(request.form['population_density'])

        features = np.array([[temperature, humidity, pm25, pm10, no2, so2, co, 
                              proximity_to_industrial_areas, population_density]])

        prediction = model.predict(features)[0]

        #numpy arrays are not hashable, so it cannot be passed as keys for the result
        #prediction = prediction.item()


        air_quality_mapping = {
            0: 'Good',
            1: 'Moderate',
            2: 'Poor',
            3: 'Hazardous'
        }
        result = air_quality_mapping.get(prediction, "Unknown")

        return render_template('res.html', result=result)

    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
