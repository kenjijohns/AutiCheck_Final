from flask_sqlalchemy import SQLAlchemy
from flask import Flask

db = SQLAlchemy()

app = Flask(__name__, template_folder='templates')
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root@localhost:3306/asd_dataset'

# Initialize the Flask app with the SQLAlchemy instance
db.init_app(app)
    
# Create a model for your data
class toddler_asd_dataset(db.Model):
    __tablename__ = 'toddler_asd_dataset'
    Case_No = db.Column(db.Integer, primary_key=True, autoincrement=True)
    A1 = db.Column(db.Integer)
    A2 = db.Column(db.Integer)
    A3 = db.Column(db.Integer)
    A4 = db.Column(db.Integer)
    A5 = db.Column(db.Integer)
    A6 = db.Column(db.Integer)
    A7 = db.Column(db.Integer)
    A8 = db.Column(db.Integer)
    A9 = db.Column(db.Integer)
    A10 = db.Column(db.Integer)
    Age_Mons = db.Column(db.Integer)
    Sex = db.Column(db.String(10))
    Ethnicity = db.Column(db.String(50))
    Family_mem_with_ASD = db.Column(db.String(5), default='No')
    Class = db.Column(db.String(10))
    
    # Function to initialize the database
def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()

# Ensure that the application is run only when executed directly (not when imported)
if __name__ == '__main__':
    init_db(app)
    app.run(debug=True)
    