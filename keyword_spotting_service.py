import keras
import librosa
import numpy as np

MODEL_PATH = "models/model.h5"
NUM_SAMPLES_TO_CONSIDER = 22050

class _Keyword_Spotting_Service:

    model = None
    _mappings = [
        "right",
        "on",
        "stop",
        "up",
        "left",
        "no",
        "go",
        "down",
        "yes",
        "off"
    ]
    _instance = None

    def predict(self, file_path):
        #extract MFCCs
        MFCCs = self.preprocess(file_path) # ( # segments, # coefficients)
        #convert 2d MFCCs array into 4d array # ( # samples, # segments, # coefficients, #channels)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]
        #make prediction
        predictions = self.model.predict(MFCCs) # [ [0.1, 0.2 , 0.1, 0.6 ...] ] #10 probabilites one for each class
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword
        

    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):
        # load audio file 
        signal, sr = librosa.load(file_path)
        #ensure consinstency in the audio file length
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]
        #extract MFCCs
        MFCCs = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        return MFCCs.T

def Keyword_Spotting_service():
    #ensure that we only have one instance of KSS (create a singleton class)
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()   
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance



if __name__ == "__main__":

    kss =  Keyword_Spotting_service()
    keyword = kss.predict("test/2.wav")
    print(f"Predicted keyword: {keyword}")