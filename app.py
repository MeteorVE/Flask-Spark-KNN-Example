from flask import Flask, render_template, url_for, jsonify, request
#import test
import knn_spark
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:admin@localhost:3306/traindb"
db = SQLAlchemy(app)

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
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def translate_text():
    data = request.get_json()
    print('[debug]:', data)
    file_url = data['url']
    num_fields = data['field']
    num_neigbour = data['neigbour']
    distance_func = data['distance']
    print(file_url, num_fields, num_neigbour, distance_func)
    response = knn_spark.KNN(file_url, int(num_fields), int(num_neigbour), distance_func)
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
	app.run(host='0.0.0.0')
	
	#print(type(o.rid), type(o.distance), type(o.score), type(o.neighbor), type(o.datasetName), type(o.featureLen), type(o.timestamp))
	
