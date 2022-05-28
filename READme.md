# Machine Learning Sing Language Recognition Application
This project focuses on the implementing the developed Machine Learning model in an OpenCV webcam application.  For each video frame, landmark information is gathered from Mediapipe Holistic and fed to model to generate a prediction. These predictions will overlaid on screen, and printed in console.

## Files:
* main.py: has the camera logic
* mp_utils.py: Contains the logic for managing Mediapipe Holistic and the data it produces
* actions.h5: File containing the trained model


## Installation and Set up
Having an updated version of Python installed on the computer is necessary for the project to work
1. Run the command ```pip install pipenv``` in your command line terminal
2. Download code from the repository
3. Go to the project location in the command line and run the command ```pipenv install --ignore-pipfile``` to install dependencies named in the pipfile.lock document
4. Run the command ```pipenv shell``` to start the virtual environment

## Running the application
You are now ready to run the sign language interpretation application.  To do so, run the command ```python main.py```.  Doing so will open a window with the camera feed for the computer.  You can begin signing and the program will interpret the signs every 30 frames. 

## NOTE:
* If the application crashes because no camera is found, it means your video capture device id used to open the camera is different from what is used in the application.  This usually happens when you are using an external web camera.  The fix for this is straightforward.  In line 16 of main.py, change the *0* within cv2.VideoCapture() to another value. Try different positive integer values until your video device is recognized.