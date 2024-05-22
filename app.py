from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, request
from database import db, toddler_asd_dataset, init_db
from flask import Flask, render_template, request, redirect, session
from flask import url_for
from flask import session
import joblib
from keras.models import load_model
from flask import Flask, render_template, request, g



# Load the trained model
model = load_model("asd_model.h5")
with open('scaler.joblib', 'rb') as file:
    scaler = joblib.load(file)
    # Load the encoder object
with open('encoder.joblib', 'rb') as file:
    encoder = joblib.load(file)
with open('model_columns.joblib', 'rb') as file:
    model_columns = joblib.load(file)
    
# Create the Flask app
app = Flask(__name__, template_folder='templates')


app.secret_key = 'Proverbs2711'

# Configure the Flask app with the SQLAlchemy instance
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://toddlerASD_aidwhatmix:25b540cda0c6795eda70e2127651638af1c1174c@05j.h.filess.io:3307/toddlerASD_aidwhatmix'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # To suppress a warning

# Initialize the Flask app with the SQLAlchemy instance
db.init_app(app)


new_data = pd.DataFrame(columns=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 
                                 'A8', 'A9', 'A10', 'Age_Mons','Sex', 'Ethnicity',
                                 'Family_mem_with_ASD'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preliminary1', methods=['GET', 'POST'])
def preliminary1():
    if request.method == 'POST':
        # Get form inputs
        sex = request.form['gender']
        ethnicity = request.form['ethnicity']
        birthday = datetime.strptime(request.form.get('birthday'), '%Y-%m-%d').date()
        present_date = datetime.now()
        age_in_months = (present_date.year - birthday.year) * 12 + (present_date.month - birthday.month)
        
          
        # Populate new_data DataFrame
        new_data.loc[0] = [None] * len(new_data.columns)
        new_data.loc[0, ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Age_Mons', 'Sex', 'Ethnicity', 'Family_mem_with_ASD']] = [None, None, None, None, None, None, None, None, None, None, age_in_months, sex, ethnicity, None]
        

        
        return render_template('preliminary2.html', new_data=new_data)
    
    return render_template('preliminary1.html')

@app.route('/preliminary2', methods=['GET', 'POST'])
def preliminary2():
    if request.method == 'POST':
        # Get form inputs
        Family_mem_with_ASD = request.form["family_diagnosed"]
        who_completed_the_test = request.form["who_completed_the_test"]
        
        # Update new_data DataFrame
        new_data.loc[0, 'Family_mem_with_ASD'] = Family_mem_with_ASD

        
    
        
        return render_template('q1.html', new_data=new_data)
    
    return render_template('preliminary2.html')


@app.route('/q1', methods=['GET', 'POST'])
def q1():
    if request.method == 'POST':
        A1 = request.form.get('answer1')
        
        if not A1:
            error_message = "Please select an answer for Question 1."
            return render_template('q1.html', new_data=new_data, error_message=error_message)
        
        new_data.loc[0, 'A1'] = int(A1)
        
        return render_template('q2.html', new_data=new_data)
    
    return render_template('q1.html')

# Add routes and functions for the remaining questions (question2, question3, ..., question10)

@app.route('/q2', methods=['GET', 'POST'])
def q2():
    if request.method == 'POST':
        A2 = request.form.get('answer2')
        
        if not A2:
            error_message = "Please select an answer for Question 2."
            return render_template('q2.html', new_data=new_data, error_message=error_message)
        
        new_data.loc[0, 'A2'] = int(A2)
      
        return render_template('q3.html', new_data=new_data)
    
    return render_template('q2.html')

@app.route('/q3', methods=['GET', 'POST'])
def q3():
    if request.method == 'POST':
        A3 = int(request.form['answer3'])
        
        new_data.loc[0, 'A3'] = A3
        return render_template('q4.html', new_data=new_data)
    return render_template('q3.html')


@app.route('/q4', methods=['GET', 'POST'])
def q4():
    if request.method == 'POST':
        A4 = int(request.form['answer4'])
        
        new_data.loc[0, 'A4'] = A4
        return render_template('q5.html', new_data=new_data)
    return render_template('q4.html')

@app.route('/q5', methods=['GET', 'POST'])
def q5():
    if request.method == 'POST':
        A5 = int(request.form['answer5'])
        
        new_data.loc[0, 'A5'] = A5
        return render_template('q6.html', new_data=new_data)
    return render_template('q5.html')

@app.route('/q6', methods=['GET', 'POST'])
def q6():
    if request.method == 'POST':
       
        A6 = int(request.form['answer6'])
        
        new_data.loc[0, 'A6'] = A6
        return render_template('q7.html', new_data=new_data)
    return render_template('q6.html')

@app.route('/q7', methods=['GET', 'POST'])
def q7():
    if request.method == 'POST':
        A7 = int(request.form['answer7'])
        
        new_data.loc[0, 'A7'] = A7
        return render_template('q8.html', new_data=new_data)
    return render_template('q7.html')

@app.route('/q8', methods=['GET', 'POST'])
def q8():
    if request.method == 'POST':
        A8 = int(request.form['answer8'])
        
        new_data.loc[0, 'A8'] = A8

        return render_template('q9.html', new_data=new_data)
    return render_template('q8.html')

@app.route('/q9', methods=['GET', 'POST'])
def q9():
    if request.method == 'POST':
        A9 = int(request.form['answer9'])
        
        new_data.loc[0, 'A9'] = A9
        return render_template('q10.html', new_data=new_data)
    return render_template('q9.html')

def predict(new_data):
    # Perform the prediction
    age_bins = np.arange(12, 37, 3)
    age_labels = np.arange(12, 36, 3)

    # Preprocess the new data
    new_data_encoded = pd.get_dummies(new_data, columns=['Sex', 'Ethnicity', 'Family_mem_with_ASD'])
    new_data['Age_Mons'] = new_data['Age_Mons'].astype(int)  # Convert Age_Mons to int

    # Concatenate numerical and encoded categorical features
    new_data_encoded['Age_binned'] = pd.cut(new_data['Age_Mons'], bins=age_bins.astype(int), labels=age_labels.astype(int))
    # Ensure that the new data has the same set of features as the training data
    missing_cols = set(model_columns) - set(new_data_encoded.columns)
    for col in missing_cols:
        new_data_encoded[col] = 0

    # Reorder columns to match training data
    new_data_processed = new_data_encoded[model_columns]
    # Normalize feature data
    new_data_scaled = scaler.transform(new_data_processed)
    new_data_3d = new_data_scaled.reshape(new_data_scaled.shape[0], new_data_scaled.shape[1], 1)
    # Make predictions on the new data
    predictions = model.predict(new_data_3d)
    return  (predictions)

@app.route('/q10', methods=['GET', 'POST'])
def q10():
    global new_data
    if request.method == 'POST':
        A10 = int(request.form['answer10'])
        
        new_data.loc[0, 'A10'] = A10
        # Extract relevant variables from the new_data DataFrame
        sex = new_data.loc[0, 'Sex']
        ethnicity = new_data.loc[0, 'Ethnicity']
        age_in_months = new_data.loc[0, 'Age_Mons']
        family_mem_with_asd = new_data.loc[0, 'Family_mem_with_ASD']
    
        result = predict(new_data)
        # Store result in session
        session['result'] = result.tolist()
        return redirect(url_for('diagnose', sex=sex, ethnicity=ethnicity, age_in_months=age_in_months, family_mem_with_asd=family_mem_with_asd))

    return render_template('q10.html')

@app.route('/diagnose', methods=['GET', 'POST'])
def diagnose():
    if request.method == 'POST':
        diagnosed_or_not = request.form.get('diagnosed')
        result= session.get('result', None)
        result_value = result[0][0]
        
        if diagnosed_or_not == 'A':
            save_to_database(new_data, result_value)
            if result_value < 0.5:
                return render_template('resultno.html',sex=request.args.get('sex'), ethnicity=request.args.get('ethnicity'), age_in_months=request.args.get('age_in_months'), family_mem_with_asd=request.args.get('family_mem_with_asd'))
            elif result_value > 0.5:
                return render_template('resultyes.html',sex=request.args.get('sex'), ethnicity=request.args.get('ethnicity'), age_in_months=request.args.get('age_in_months'), family_mem_with_asd=request.args.get('family_mem_with_asd'))
        elif diagnosed_or_not == 'B':
            # Save to the database with class "no"
            save_to_database(new_data, 0)  # Assuming class "no" corresponds to 0
            if result_value < 0.5:
                return render_template('resultno.html',sex=request.args.get('sex'), ethnicity=request.args.get('ethnicity'), age_in_months=request.args.get('age_in_months'), family_mem_with_asd=request.args.get('family_mem_with_asd'))
            elif result_value > 0.5:
                return render_template('resultyes.html',sex=request.args.get('sex'), ethnicity=request.args.get('ethnicity'), age_in_months=request.args.get('age_in_months'), family_mem_with_asd=request.args.get('family_mem_with_asd'))
        elif diagnosed_or_not == 'C':
            # Save to the database with class "yes"
            save_to_database(new_data, 1)  # Assuming class "yes" corresponds to 1
            if result_value < 0.5:
                return render_template('resultno.html',sex=request.args.get('sex'), ethnicity=request.args.get('ethnicity'), age_in_months=request.args.get('age_in_months'), family_mem_with_asd=request.args.get('family_mem_with_asd'))
            elif result[0] > 0.5:
                return render_template('resultyes.html',sex=request.args.get('sex'), ethnicity=request.args.get('ethnicity'), age_in_months=request.args.get('age_in_months'), family_mem_with_asd=request.args.get('family_mem_with_asd'))
        else:
            pass

    # Handle GET request (if needed)
    return render_template('diagnose.html')



def save_to_database(new_data, result_value):
    # Create a new CaseData instance with the data
    case_data = toddler_asd_dataset(
        A1=new_data.loc[0, 'A1'],
        A2=new_data.loc[0, 'A2'],
        A3=new_data.loc[0, 'A3'],
        A4=new_data.loc[0, 'A4'],
        A5=new_data.loc[0, 'A5'],
        A6=new_data.loc[0, 'A6'],
        A7=new_data.loc[0, 'A7'],
        A8=new_data.loc[0, 'A8'],
        A9=new_data.loc[0, 'A9'],
        A10=new_data.loc[0, 'A10'],
        Age_Mons=new_data.loc[0, 'Age_Mons'],
        Sex=new_data.loc[0, 'Sex'],
        Ethnicity=new_data.loc[0, 'Ethnicity'],
        Family_mem_with_ASD=new_data.loc[0, 'Family_mem_with_ASD'],
        Class="Yes" if result_value > 0.5 else "No"
    )

    # Add the instance to the session and commit changes to the database
    db.session.add(case_data)
    db.session.commit()
    db.session.remove()

@app.route('/about_asd')
def about_asd():
    return render_template('about_asd.html')
@app.route('/paper')
def paper():
    return render_template('paper.html')

    
# Ensure that the application is run only when executed directly (not when imported)
if __name__ == '__main__':
    app.run(debug=True)
  