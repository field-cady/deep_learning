from flask import Flask, request, jsonify

app = Flask(__name__)

# Define an endpoint for calling the predict function based on your ml library/framework
@app.route("/predict", methods=["GET","POST"])
def predict():
    # Load the Input
    data = request.get_json(force=True)#.decode('utf-8')
    #data = "foo"
    
    # Load the model
    #model = load_model()
    
    # Make predictions on input data
    #model.predict(data) # .predict() could change based on libarary/framework
    return '-' + data['username'] + '_output'
  
  
# Start the flask app and allow remote connections
app.run(host='0.0.0.0', port = 80)