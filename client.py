import requests

#url to Flask development server (running at 5000)
#URL = "http://127.0.0.1:5000/predict"
#url to uwsgi web server running at 5050
URL = "http://127.0.0.1:5050/predict"
TEST_AUDIO_FILE_PATH = "test/2.wav"


if __name__ == "__main__": 
    print("hey")
    audio_file = open(TEST_AUDIO_FILE_PATH, "rb")
    values = {"file": (TEST_AUDIO_FILE_PATH, audio_file, "audio/wav")}
    response = requests.post(URL, files = values)
    data = response.json()

    print(f"Predicted keyword is: {data['keyword']}")