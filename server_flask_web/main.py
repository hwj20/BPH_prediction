# encoding=utf-8
# TODO: 登录注册，选择模型，上传和管理数据
import hashlib

from flask import render_template, request, Flask, jsonify
import data_manager

# risk factor
# Index(['age', 'drink_state', 'RBCs', 'LDL', 'smoke_state', 'GHb_A1c',
#        'PLT', 'has_surgery', 'WBC', 'FPSA', 'PCV', 'is_BPH'],
#       dtype='object')

from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
# sample db
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost:3306/BPH_login'
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

    def __repr__(self):
        return '<User %r>' % self.username


@app.route('/')
def index():
    return render_template('index.html')


# 吐槽：我不想写用户id冲突部分的检测了，这个只能保证，你的id是hash出来的，几乎不可能会撞; 要是id撞了就会报错
@app.route('/user/register', methods=['POST'])
def handle_register():
    file = request.files['file']
    user = request.form['user']
    user = eval(user)

    username = user['username']
    password = user['password']
    md5 = hashlib.md5()
    md5.update(username.encode('utf-8'))
    md5.update(password.encode('utf-8'))
    id = int(md5.hexdigest(), 16) % (10 ** 8)
    new_user = User(username=username, password=password, id=id)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'flag': True, 'data': 'success', 'id': new_user.id})


@app.route('/user/login', methods=['POST'])
def login():
    # Get the form data from the request
    form_data = request.json

    flag = True  # replace with your actual flag
    message = 'Success'  # replace with your actual message
    user = User.query.filter_by(username=form_data['username'], password=form_data['password']).first()
    if user:
        message = 'Success'  # replace with your actual message
    else:
        message = 'Failed'
        flag = False

    # Return a JSON response with the flag and message variables
    response_data = {
        'flag': flag,
        'data': message
    }
    return jsonify(response_data)


@app.route('/model/select/<string:id>', methods=['POST'])
def model_select(id):
    # model = Model()
    # results = model.findSome(**query_params)
    model_name = request.path.split('/')[-1]
    with open('../checkpoints/selected_model.txt', 'w') as f:
        f.write(model_name)
    print(model_name)
    return 'Success', 200


# TODO
@app.route('/model/findSome', methods=['GET'])
def find_some():
    # model = Model()
    query_params = request.args.to_dict()
    print(query_params)
    # results = model.findSome(**query_params)
    # response_data = [data_manager.get_all_model()[2]]
    response_data = data_manager.get_all_model()

    return jsonify(data=response_data)


@app.route('/model/findAll', methods=['GET'])
def find_all():
    response_data = data_manager.get_all_model()
    return jsonify(data=response_data)


@app.route('/type/findAll', methods=['GET'])
def find_all_type():
    response_data = data_manager.get_all_model(False)
    response_data = response_data['type']

    return jsonify(data=response_data)


@app.route('/user/findAll', methods=['GET'])
def find_all_user():
    response_data = data_manager.get_all_model(False)
    response_data = response_data['belongTo']
    return jsonify(data=response_data)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
