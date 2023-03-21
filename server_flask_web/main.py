from flask import render_template, request, Flask, jsonify

# risk factor
# Index(['age', 'drink_state', 'RBCs', 'LDL', 'smoke_state', 'GHb_A1c',
#        'PLT', 'has_surgery', 'WBC', 'FPSA', 'PCV', 'is_BPH'],
#       dtype='object')


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/user/login', methods=['POST'])
def login():
    # Get the form data from the request
    form_data = request.json

    # Perform the login logic and set the flag and message variables accordingly
    # For example, you could check if the user exists in the database and if the password is correct
    flag = True  # replace with your actual flag
    message = 'Success'  # replace with your actual message

    # Return a JSON response with the flag and message variables
    response_data = {
        'flag': flag,
        'data': message
    }
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
