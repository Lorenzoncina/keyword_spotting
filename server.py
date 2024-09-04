import os
import random
from flask import Flask, request, jsonify
from keyword_spotting_service import Keyword_Spotting_service

app = Flask(__name__)

#route an incoming request to an endpoint

@app.route("/predict", methods=["POST"])
def predict():

    # get audio file and save it
    audio_file = request.files["file"]
    file_name = str(random.randint(0,10000))
    audio_file.save(file_name)
    
    #invoke keyword spotting service
    kss = Keyword_Spotting_service()
   
    # make a prediction
    predicted_keyword = kss.predict(file_name)
    
    #remove the audio file
    os.remove(file_name)
    
    #send back the predicted keyword in json format
    data = {"keyword": predicted_keyword}
    return jsonify(data)



if __name__ == "__main__":
    app.run(debug=False)