from flask import Flask, render_template, url_for, jsonify, request, redirect, session, flash
import test
#import final
from datetime import timedelta
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = "hellothisismysecretkey"
app.config['JSON_AS_ASCII'] = False

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
#app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:admin@localhost:3306/traindb"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users_table.sqlite3' # "users_table" here is the name of the table that you're gonna be referencing 
app.permanent_session_lifetime = timedelta(minutes=3)
db = SQLAlchemy(app)


class users_table(db.Model): # The columns represent pieces of information；Rows represent in ；Rows represent individual items
    _id = db.Column("id",db.Integer, primary_key=True) # id will be automatically be created for us because it's a primary key
    name = db.Column(db.String(100)) # 100 here is the maximum length of the string that we want to store(100 characters)
    email = db.Column(db.String(100)) # string也可以改成integer/float/boolean

    def __init__(self,name,email): # We want to store users and each users has a name and an email (these 2 are what we need every time we define a new user object)(the init method will take the variables that we need to create a new object)
        self.name = name
        self.email = email

class Knn(db.Model):
    rid = db.Column(db.Integer, primary_key=True)
    distance = db.Column(db.VARCHAR(255))
    score = db.Column(db.Float)
    neighbor = db.Column(db.Integer)
    datasetName = db.Column(db.VARCHAR(255))
    featureLen = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    
    def __init__(self, distance='', score=0, neighbor=1, datasetName='', featureLen=0):
        self.distance = distance
        self.score = score
        self.neighbor = neighbor
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
def knn_algo():
    return render_template('knn.html')

@app.route("/login",methods=["POST","GET"])
def login():
    if request.method == "POST":
        session.permanent = True #used to define this specific session as a permanent session which means it's gonna last as long as we define up there 
        user = request.form["nm"]
        session["user"] = user
        found_user = users_table.query.filter_by(name=user).first()
        if found_user: # When an user types his name, we'll check if this user is already exist. If not then we'll create one
            session["email"] = found_user.email
        else:
            usr = users_table(user, "")
            db.session.add(usr) # add this user model to our database
            db.session.commit()

        flash("Login Succesful!")
        return redirect(url_for("user"))
    else:
        if "user" in session: #代表若已經是signed in的狀態
            flash("Already Logged in!")
            return redirect(url_for("user"))
        return render_template("login.html")

@app.route("/user",methods=["POST","GET"])
def user():
    email = None
    if "user" in session:
        user = session["user"]
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
        return render_template("User.html", email=email)
    else:
        flash("You are not logged in!")
        return redirect(url_for("login"))

@app.route("/logout")
def logout():
    #if "user" in session:
    #user = session["user"]
    flash("You have been logged out!", "info")
    session.pop("user",None) #remove the user data from my session 
    session.pop("email",None)
    return redirect(url_for("login"))

@app.route("/view")
def view():
    return render_template("view.html",values=users_table.query.all())

@app.route("/About")
def About():
    return render_template("about.html")

@app.route('/train', methods=['POST'])
def translate_text():
    data = request.get_json()
    print('[debug]:', data)
    file_url = data['url']
    num_fields = data['field']
    num_neigbour = data['neigbour']
    distance_func = data['distance']
    print(file_url, num_fields, num_neigbour, distance_func)
    response = test.KNN(file_url, int(num_fields), int(num_neigbour), distance_func)
    p = Knn(response[0], response[1], response[2], response[3], response[4])
    p.save_to_db()
    print(response)
    return jsonify(response[1])

@app.route('/queryall', methods=['POST'])
def query_all_data():
    data = request.get_json()
    print(data['query_num'])
    response = {}
    for idx,o in enumerate(Knn.query.all()):
        response[idx] = [o.rid, o.distance, o.score, o.neighbor, o.datasetName, o.featureLen, o.timestamp.strftime("%m/%d/%Y, %H:%M:%S")]
        if idx == data['query_num']:
        	break
    print(response)	
    return jsonify(response)



if __name__ == '__main__':
    #db.create_all()
    #app.run(host='0.0.0.0')
    app.run(debug=True)
    db.create_all()
	#print(type(o.rid), type(o.distance), type(o.score), type(o.neighbor), type(o.datasetName), type(o.featureLen), type(o.timestamp))
	
