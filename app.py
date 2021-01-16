from flask import Flask, render_template, url_for, jsonify, request, redirect, session, flash
import test
from datetime import timedelta
from flask_sqlalchemy import SQLAlchemy
import configparser
import json
#import knn_spark

app = Flask(__name__)
app.secret_key = "hellothisismysecretkey"
app.config['JSON_AS_ASCII'] = False
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
#app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:admin@localhost:3306/traindb"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users_table.sqlite3' # "users_table" here is the name of the table that you're gonna be referencing 
app.permanent_session_lifetime = timedelta(minutes=3)
db = SQLAlchemy(app)

from google.oauth2 import id_token
import google.auth.transport.requests as google_request
import requests
config = configparser.ConfigParser()
config.read('Config.ini')
GOOGLE_OAUTH2_CLIENT_ID = config['GOOGLE']['ClientId']

import os
from werkzeug.utils import secure_filename
UPLOAD_FOLDER = './upload'
ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg', 'gif', 'txt'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024

class users_table(db.Model): # The columns represent pieces of information；Rows represent in ；Rows represent individual items
    _id = db.Column("id",db.Integer, primary_key=True) # id will be automatically be created for us because it's a primary key
    name = db.Column(db.String(100)) # 100 here is the maximum length of the string that we want to store(100 characters)
    email = db.Column(db.String(100)) # string也可以改成integer/float/boolean
    password = db.Column(db.String(100))

    def __init__(self, name, password, email): # We want to store users and each users has a name and an email (these 2 are what we need every time we define a new user object)(the init method will take the variables that we need to create a new object)
        self.name = name
        self.password = password
        self.email = email

class knn(db.Model):
    # Class Name = Table Name
    rid = db.Column(db.Integer, primary_key=True, autoincrement = True)
    distance = db.Column(db.VARCHAR(255))
    username = db.Column(db.VARCHAR(255))
    score = db.Column(db.Float)
    neighbor = db.Column(db.Integer)
    dataset_name = db.Column(db.VARCHAR(255))
    #featureLen = db.Column(db.Integer)
    seed = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    
    def __init__(self, distance_func='', username='anonymous', score=0, num_nearest_neigbours=3, dataset='', seed=10):
        self.distance = distance_func
        self.username = username
        self.score = score
        self.neighbor = num_nearest_neigbours
        self.seed = seed
        self.dataset_name = dataset
        #self.featureLen = featureLen
       
    def save_to_db(self):
        db.session.add(self) 
        db.session.commit()


class nb(db.Model):
    # Class Name = Table Name
    rid = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.VARCHAR(255))
    score = db.Column(db.Float)
    dataset_name = db.Column(db.VARCHAR(255))
    seed = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

    def __init__(self, username='anonymous', score=0, dataset='', seed=10):
        self.username = username
        self.score = score
        self.seed = seed
        self.dataset_name = dataset

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()


class lr(db.Model):
    rid = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.VARCHAR(255))
    score = db.Column(db.Float)
    dataset_name = db.Column(db.VARCHAR(255))
    seed = db.Column(db.Integer)
    iterations = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

    def __init__(self, username='anonymous', score=0, iterations=10, dataset='', seed=10):
        self.username = username
        self.score = score
        self.seed = seed
        self.dataset_name = dataset
        self.iterations = iterations

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()


class dt(db.Model):
    rid = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.VARCHAR(255))
    score = db.Column(db.Float)
    dataset_name = db.Column(db.VARCHAR(255))
    categorical_features_info = db.Column(db.VARCHAR(255))
    seed = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

    def __init__(self, username='anonymous', categoricalFeaturesInfo={}, score=0, dataset='', seed=10):
        self.username = username
        self.score = score
        self.seed = seed
        self.dataset_name = dataset
        self.categorical_features_info = json.dumps(categoricalFeaturesInfo)

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()

class rf(db.Model):
    rid = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.VARCHAR(255))
    score = db.Column(db.Float)
    dataset_name = db.Column(db.VARCHAR(255))
    categorical_features_info = db.Column(db.VARCHAR(255))
    seed = db.Column(db.Integer)
    num_tree = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

    def __init__(self, username='anonymous', categoricalFeaturesInfo={}, numTrees=5, score=0, dataset='', seed=10):
        self.username = username
        self.score = score
        self.seed = seed
        self.dataset_name = dataset
        self.num_tree = numTrees
        self.categorical_features_info = json.dumps(categoricalFeaturesInfo)

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/knn')
def knn_algo_page():
    return render_template('knn.html')

# @app.route('/knn_predict')
# def knn_predict_page():
#     return render_template('knn_predict.html')

@app.route('/nb')
def nb_algo_page():
    return render_template('nb.html')

@app.route('/lr')
def lr_algo_page():
    return render_template('lr.html')

@app.route('/dt')
def dt_algo_page():
    return render_template('dt.html')

@app.route('/rf')
def rf_algo_page():
    return render_template('rf.html')

@app.route('/nb_predict')
def nb_predict_page():
    return render_template('nb_predict.html')

@app.route('/lr_predict')
def lr_predict_page():
    return render_template('lr_predict.html')

@app.route('/dt_predict')
def dt_predict_page():
    return render_template('dt_predict.html')

@app.route('/rf_predict')
def rf_predict_page():
    return render_template('rf_predict.html')


@app.route('/upload_event', methods=["POST"])
def upload_event():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],
                                   filename))
            print("[debug in upload_event] filename=", filename, flush=True)
    return redirect(request.referrer)
    #return jsonify({}), 200
            #session['uploaded_filename'] = filename
            #return redirect(url_for('uploaded_file',
            #                        filename=filename))


@app.route("/login",methods=["POST","GET"])
def login():
    if request.method == "POST":
        session.permanent = True #used to define this specific session as a permanent session which means it's gonna last as long as we define up there 
        user = request.form["nm"]
        password = request.form["pwd"]

        found_user = users_table.query.filter_by(name=user).first()
        if found_user: # When an user types his name, we'll check if this user is already exist. If not then we'll create one
            print(found_user, flush=True)
            if found_user.password != password:
                flash("Login Failed ! plz check your email or password!")
                return redirect(url_for("login"))
            else:
                session["user"] = user
            session["email"] = found_user.email
        else:
            usr = users_table(user, password, "")
            db.session.add(usr) # add this user model to our database
            db.session.commit()

        flash("Login Succesful!")
        return redirect(url_for("user"))
    else:
        if "user" in session: #代表若已經是signed in的狀態
            flash("Already Logged in!")
            return redirect(url_for("user"))
        else:
            return render_template("login.html", google_oauth2_client_id=GOOGLE_OAUTH2_CLIENT_ID)



@app.route("/user",methods=["POST","GET"])
def user():
    email = None
    user = None
    if "user" in session:
        user = session["user"]
        print("[debug in user()]: session['user']:",
              session["user"],"; user:", user, flush=True)

        if request.method == "POST": 
            email = request.form["email"] # grab that email from the email field
            session["email"] = email # store it in the session
                   
            found_user = users_table.query.filter_by(name=user).first()
            found_user.email = email
            db.session.commit() # next time we login this will be saved
            flash("Email was saved!")
            return redirect("/")

        else: # if it's a GET request
            if "email" in session:
                email = session["email"] # get the email from the session
        return render_template("user.html", email=email, google_oauth2_client_id=GOOGLE_OAUTH2_CLIENT_ID)
    else:
        flash("You are not logged in!")
        return redirect(url_for("login"))

@app.route("/logout")
def logout():
    #if "user" in session:
    #user = session["user"]
    if "google_token" in session:
        # print("[debug in logout] token=", session['google_token'], flush=True)
        # response = requests.post('https://accounts.google.com/o/oauth2/revoke',
        #               params={'token': session['google_token']},
        #               headers={'content-type': 'application/x-www-form-urlencoded'}) 
        # print(response.content, flush=True)
        session.pop('google_token')
    flash("You have been logged out!", "info")
    session.pop("user",None) #remove the user data from my session 
    session.pop("email",None)
    return redirect(url_for("login"))

@app.route("/view")
def view():
    return render_template("view.html",values=users_table.query.all())

@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/google_sign_in', methods=['POST'])
def google_sign_in():
    token = request.json['id_token']
    username = request.json['email']

    try:
        # Specify the GOOGLE_OAUTH2_CLIENT_ID of the app that accesses the backend:
        id_info = id_token.verify_oauth2_token(
            token,
            google_request.Request(),
            GOOGLE_OAUTH2_CLIENT_ID
        )

        if id_info['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            raise ValueError('Wrong issuer.')

        # ID token is valid. Get the user's Google Account ID from the decoded token.
        # user_id = id_info['sub']
        # reference: https://developers.google.com/identity/sign-in/web/backend-auth
    except ValueError:
        # Invalid token
        raise ValueError('Invalid token')

    print('Login Success', username, "\nToken:\n", token, flush=True)
    flash("Login Succesful!")
    session['user'] = username
    session['google_token'] = token

    #return redirect("/user")
    return jsonify({}), 200


import train_with_class

@app.route('/train', methods=['POST'])
def train_with_algo():
    data = request.get_json()
    algorithm = data.pop(
        'algorithm', '[in train_with_algo()] error, unknown algorithm  ...')
    # if use_algo == 'KNN':
    #     request_dict['dataset'], request_dict['num_fields, request_dict['num_neigbour, request_dict['distance_func, request_dict['seed = data['dataset'], data[
    #         'field'], data['neigbour'], data['distance'], data['seed']
    # elif use_algo == 'LR':
    #     dataset, seed, iterations = data['dataset'], data[
    #         'seed'], data['iterations']
    # elif use_algo == 'DT':
    #     dataset, seed, categoricalFeaturesInfo = data['dataset'], data[
    #         'seed'], json.loads(data['categoricalFeaturesInfo'])
    # elif use_algo == 'RF':
    #     dataset, seed, categoricalFeaturesInfo, numTrees = data['dataset'], data[
    #         'seed'], json.loads(data['categoricalFeaturesInfo']), data['numTrees']
    
    # Run Training process
    train_score = train_with_class.call_train_function(
        algorithm=algorithm, mode='train', algorithm_parameter=data)  # call_train_function(algorithm, mode, algorithm_parameter)

    print('[debug in train_with_algo()]: data=', data, flush=True)
    print('[debug in train_with_algo()]: response=', train_score, flush=True)
    
    # Create a record and Save to DB
    p = globals()[algorithm.lower()](
        **data, username='anonymous' if "user" not in session else session['user'])
    p.save_to_db()
    
    return jsonify(train_score)


@app.route('/predict', methods=['POST'])
def predict_with_algo():
    data = request.get_json()
    algorithm = data.pop(
        'algorithm', '[in train_with_algo()] error, unknown algorithm  ...')

    # Run Predicting process
    train_score = train_with_class.call_train_function(
        algorithm=algorithm, mode='test', algorithm_parameter=data)  # call_train_function(algorithm, mode, algorithm_parameter)

    print('[debug in train_with_algo()]: data=', data, flush=True)
    print('[debug in train_with_algo()]: response=', train_score, flush=True)

    return jsonify(train_score)


@app.route('/query', methods=['POST'])
def query_train_data():
    data = request.get_json()
    print("[debug in query_train_data()] data=", data, flush=True)

    db_data = None
    if data['query_num'] == '':
        data['query_num'] = 10

    algo_class = globals()[data['use_algo'].lower()]
    if data['query_table_key'] == "all":
        db_data = algo_class.query.limit(int(data['query_num']))
    elif data['query_table_key'] == "distance":
        db_data = algo_class.query.filter_by(
            distance=data['query_table_value']).limit(int(data['query_num']))
    elif data['query_table_key'] == "datasetName":
        db_data = algo_class.query.filter_by(
            dataset_name=data['query_table_value']).limit(int(data['query_num']))
    elif data['query_table_key'] == "neighbor":
        db_data = algo_class.query.filter_by(
            neighbor=data['query_table_value']).limit(int(data['query_num']))
    elif data['query_table_key'] == "seed":
        db_data = algo_class.query.filter_by(
            seed=data['query_table_value']).limit(int(data['query_num']))
    elif data['query_table_key'] == "iterations":
        db_data = algo_class.query.filter_by(
            iterations=data['query_table_value']).limit(int(data['query_num']))
    elif data['query_table_key'] == "categorical_features_info":
        db_data = algo_class.query.filter_by(
            categorical_features_info=data['query_table_value']).limit(int(data['query_num']))
    elif data['query_table_key'] == "num_tree":
        db_data = algo_class.query.filter_by(
            num_tree=data['query_table_value']).limit(int(data['query_num']))
    else:
        db_data = ["key not found"]

    response = {}
    if data['use_algo'] == 'KNN':
        for idx, o in enumerate(db_data):
            response[idx] = [o.rid, o.dataset_name, o.distance, o.score, o.neighbor,
                         o.seed, o.timestamp.strftime("%m/%d/%Y, %H:%M:%S")]
    elif data['use_algo'] == 'NB':
        for idx, o in enumerate(db_data):
            response[idx] = [o.rid, o.dataset_name, o.score, o.seed,
                             o.timestamp.strftime("%m/%d/%Y, %H:%M:%S")]
    elif data['use_algo'] == 'LR':
        for idx, o in enumerate(db_data):
            response[idx] = [o.rid, o.dataset_name, o.score, o.iterations, o.seed,
                             o.timestamp.strftime("%m/%d/%Y, %H:%M:%S")]
    elif data['use_algo'] == 'DT':
        for idx, o in enumerate(db_data):
            response[idx] = [o.rid, o.dataset_name, o.score, o.categorical_features_info, o.seed,
                             o.timestamp.strftime("%m/%d/%Y, %H:%M:%S")]
    elif data['use_algo'] == 'RT':
        for idx, o in enumerate(db_data):
            response[idx] = [o.rid, o.dataset_name, o.score, o.categorical_features_info, o.num_tree, o.seed,
                             o.timestamp.strftime("%m/%d/%Y, %H:%M:%S")]
    response = jsonify(status="success", data=response)
    print(response, flush=True)
    return response


def allowed_file(filename):
  return '.' in filename and \
    filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS




if __name__ == '__main__':
    #db.create_all()
    #app.run(host='0.0.0.0')
    app.run(debug=True, port=5000, host='0.0.0.0')
    db.create_all()
	#print(type(o.rid), type(o.distance), type(o.score), type(o.neighbor), type(o.datasetName), type(o.featureLen), type(o.timestamp))
	
