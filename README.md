# Python Flask for Show and Tell Inference Mode

This application runs the show and tell pretained model and serves the inference requests.

## Run the app locally

1. [Install Python][]
1. cd into this project's root directory
1. Run `pip install -r requirements.txt` to install the app's dependencies
1. Run `python concat_chkp.py` to concatenate the checkpoint chunks
1. Run `python app.py`
1. Access the running app in a browser at <http://localhost:5000>
1. The default ID/Passwd is admin/time4fun

[Install Python]: https://www.python.org/downloads/

## Run the app in IBM Cloud

1. Log into IBM Cloud
1. Create a Python Flask app with a unique app name and a unique host name
1. Go to your workstation, [Install Python][]
1. cd into this project's root directory
1. Run `pip install -r requirements.txt` to install the app's dependencies
1. Run `python concat_chkp.py` to concatenate the checkpoint chunks
1. Update `manifest.yml` with your app name and host name
1. Upload the app to IBM Cloud, as described in <https://console.bluemix.net/docs/starters/upload_app.html>
1. Access the running app in a browser at <https://yourhostname.mybluemix.net/>
1. The default ID/Passwd is admin/time4fun
