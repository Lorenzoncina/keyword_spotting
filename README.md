# keyword_spotting
Identification of keyword in audio files


Step of th eproject:

1 - preprocessing data
2 - training a CNN for keyword spotting
3 - wrapping the trained model into a inference script (or service called keyword spotting, which is a singleton class)
4 - structure the application as a client-server application with Flask (which is not though a production server)
5 - install and run uWSGI web serever to run the server 


To run uWSGI server:
 uwsgi --http 127.0.0.1:5050 --wsgi-file server.py --callable app --processes 1 --nthreads 1

Otherwise we can create the configuration file app.ini which save all these parameter required to run uwsgi web server
uwsgi app.ini
