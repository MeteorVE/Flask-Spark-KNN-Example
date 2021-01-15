from flask import Flask, render_template, url_for, jsonify, request, redirect, session, flash
import test
#import final
from datetime import timedelta
from flask_sqlalchemy import SQLAlchemy
import configparser

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

from werkzeug.utils import secure_filename
UPLOAD_FOLDER = 'C:/Users/MeteorV/Desktop/upload'
ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

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
    datasetName = db.Column(db.VARCHAR(255))
    featureLen = db.Column(db.Integer)
    seed = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    
    def __init__(self, distance='', username='anonymous', score=0, neighbor=1, datasetName='', featureLen=0, seed=10):
        self.distance = distance
        self.username = username
        self.score = score
        self.neighbor = neighbor
        self.seed = seed
        self.datasetName = datasetName
        self.featureLen = featureLen
       

    def save_to_db(self):
        db.session.add(self) 
        db.session.commit()

    def query_all(self):
        #self.query.filter_by(rid= _rid).first()
        self.query.all()


@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/knn')
def knn_algo_page():
    return render_template('knn.html')

@app.route('/knn_predict')
def knn_predict_page():
    return render_template('knn_predict.html')



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

@app.route('/train', methods=['POST'])
def train_with_algo():
    data = request.get_json()
    print('[debug in train_with_algo()]: data=', data, flush=True)
    file_url = data['url']
    num_fields = data['field']
    num_neigbour = data['neigbour']
    distance_func = data['distance']
    seed = data['seed']
    response = test.KNN(file_url, int(num_fields), int(num_neigbour), distance_func)
    print('[debug in train_with_algo()]: response=', response, flush=True)
    p = knn(distance=response[0], score=response[1],
            neighbor=response[2], datasetName=response[3], featureLen=response[4], username='anonymous' if "user" not in session else session['user'])
    p.save_to_db()
    return jsonify(response[1])


@app.route('/query', methods=['POST'])
def query_train_data():
    data = request.get_json()
    print("[debug in query_train_data()] data=", data, flush=True)

    db_data = None
    if data['query_num'] == '':
        data['query_num'] = 10

    algo_class = globals()[data['use_algo']]
    if data['query_table_key'] == "all":
        db_data = algo_class.query.limit(int(data['query_num']))
    elif data['query_table_key'] == "distance":
        db_data = algo_class.query.filter_by(
            distance=data['query_table_value']).limit(int(data['query_num']))
    elif data['query_table_key'] == "datasetName":
        db_data = algo_class.query.filter_by(
            datasetName=data['query_table_value']).limit(int(data['query_num']))
    elif data['query_table_key'] == "neighbor":
        db_data = algo_class.query.filter_by(
            neighbor=data['query_table_value']).limit(int(data['query_num']))
    elif data['query_table_key'] == "seed":
        db_data = algo_class.query.filter_by(
            seed=data['query_table_value']).limit(int(data['query_num']))
    else:
        db_data = ["key not found"]

    response = {}
    for idx, o in enumerate(db_data):
        response[idx] = [o.rid, o.distance, o.score, o.neighbor, o.datasetName,
                         o.featureLen, o.seed, o.timestamp.strftime("%m/%d/%Y, %H:%M:%S")]
        if idx == data['query_num']:
            break
    response = jsonify(status="success", data=response)
    print(response, flush=True)
    return response


def allowed_file(filename):
  return '.' in filename and \
    filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS




if __name__ == '__main__':
    #db.create_all()
    #app.run(host='0.0.0.0')
    app.run(debug=True)
    db.create_all()
	#print(type(o.rid), type(o.distance), type(o.score), type(o.neighbor), type(o.datasetName), type(o.featureLen), type(o.timestamp))
	
