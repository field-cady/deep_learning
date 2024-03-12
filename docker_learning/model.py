from flask import Flask, request, jsonify, session
from flask_session import Session

app = Flask(__name__)
app.secret_key = 'foo'
Session(app)

#n = 0

# Define an endpoint for calling the predict function based on your ml library/framework
@app.route("/predict", methods=["GET","POST"])
def predict():
    # Load the Input
    data = request.get_json(force=True)
    
    # Load the model
    #model = load_model()
    
    return f"{data['username']} has hit\n"
  
  
# Start the flask app and allow remote connections
app.run(host='0.0.0.0', port = 80)