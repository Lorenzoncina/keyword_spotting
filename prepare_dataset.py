import os
import json
import librosa


#dowload the data 
"""
save_path = "/home/concina/machine_learning_python_book/keyword_spotting"
dataset= load_dataset("google/speech_commands", 'v0.01', cache_dir=save_path)
"""

#some handy constants
DATASET_PATH = "/home/concina/machine_learning_python_book/keyword_spotting/google_speech_command_data"
JSON_PATH = "data.json"
SAMPLE_TO_CONSIDER = 22050 #1 second worth of sound (22050 sr)

#go through all the audio file and extract MFFCC, store in the output JSON file
def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):
    #data dictionary which will be saved as an output json file
    data = {
        "mappings" : [],
        "labels" : [],
        "MFCCs" : [],
        "files": []
    }

    #loop through all the sub-dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        #we need to ensure that we are not at root level
        if dirpath is not dataset_path:
            #update mappings    
            category = dirpath.split("/")[-1]
            data['mappings'].append(category)
            print(f"Processing {category}")

            # loop through all the filenames and extract MFCCs
            for f in filenames:
                #get the filepath
                file_path = os.path.join(dirpath, f)
                #load the audio file
                signal, sr = librosa.load(file_path)
                #ensure the audio file is at least 1 sec (ignore shorter ones)
                if len(signal) >= SAMPLE_TO_CONSIDER:
                    #enforce  1 sec long signal
                    signal = signal[:SAMPLE_TO_CONSIDER]
                    #extract the MFCCs
                    MFCCs = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
                    #store data
                    data["labels"].append(i-1)
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["files"].append(file_path)
                    print(f"{file_path}: {i-1}")
    
    #store in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, JSON_PATH)


