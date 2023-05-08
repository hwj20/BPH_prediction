from flask import render_template, request, Flask

# risk factor
# Index(['age', 'drink_state', 'RBCs', 'LDL', 'smoke_state', 'GHb_A1c',
#        'PLT', 'has_surgery', 'WBC', 'FPSA', 'PCV', 'is_BPH'],
#       dtype='object')


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/calculate', methods=['POST'])
def calculate():
    # Get input values from the form
    age = request.form['age']
    fpsa = request.form['fpsa']
    drink_state = request.form['drink_state']
    RBCs = request.form['RBCs']
    LDL = request.form['LDL']
    smoke_state = request.form['smoke_state']
    GHb_A1c = request.form['GHb_A1c']
    PLT = request.form['PLT']
    has_surgery = request.form['has_surgery']
    WBC = request.form['WBC']
    PCV = request.form['PCV']
    # is_BPH = request.form['is_BPH']
    import pickle

    # Convert input values to a feature vector
    X_test = [[float(age), float(fpsa), float(drink_state), float(RBCs), float(LDL),
               float(smoke_state), float(GHb_A1c),
               float(PLT), float(has_surgery), float(WBC), float(PCV)]]
    # float(is_BPH)]]

    model_name = 'DecisionTreemodel.pkl'
    with open('../checkpoints/selected_model.txt', 'r') as f:
        model_name = f.read()
    # Load the saved SVM model
    with open('../checkpoints/' + model_name, 'rb') as f:
        model = pickle.load(f)
        # Use the loaded SVM model to predict the label score
        label_score = model.predict(X_test)

    # Determine risk group based on risk score
    if label_score >= 0.5:
        risk_group = 'HIGH'
    else:
        risk_group = 'LOW'

    # Render the results page with the risk group
    return render_template('results.html', risk_group=risk_group)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
