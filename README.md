# Darts-Video-Server (Darts-Backend)

Installation instructions:
1. Download or clone this repository
2. Install Python if not done yet (developed on Python 3.11)
3. Run ```py -m venv venv``` in the root directory of this project, to setup a new virtual environment for installing the projects' dependencies.

If you're on Windows Powershell you might need step 4. to execute step 5., otherwise skip this step

4. Run ```Set-ExecutionPolicy -ExecutionPolicy RemoteSigned``` in powershell

5. Run ```.\venv\Scripts\activate.ps1``` to activate your virtual environment
6. Run ```pip install -r requirements.txt``` to install all dependencies/requirements in your new virtual environment
7. Run ```flask run --host=0.0.0.0``` to start the Flask "Darts-Video-Server"
