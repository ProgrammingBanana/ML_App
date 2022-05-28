import cv2
import numpy as np
from mp_utils import mp_utilities
from tensorflow import keras


def main():
    """ Contains the logic to handle frame data capture, model prediction and prediction visualization
    """
    # Instantiating mp_utilities object to handle Mediapipe functions
    utils = mp_utilities()
    actions = np.array(['ayuda', 'clase', 'donde', 'gracias', 'hola', 'necesitar', 'no entender', 'repetir', 'n-a', 'empty'])
    # opening machine learning model from 'actions.h5' file
    model = keras.models.load_model("actions.h5")
    # Array to contain accumulated frame data
    sequence = [] # Append data over 30 frames
    # String for prediction presentation
    word = "" # Allows us to concatenate list of predictions
    threshold = 0.45 # Confidence metric (only render results if it meets our threshold value)

    # Connects to the webcam (the number can vary depending on machine)
    cap = cv2.VideoCapture(0)

    # Set mediapipe model
    with utils.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # While the camera is opened
        while cap.isOpened():
            
            # Read feed
            ret, frame = cap.read()
            
            # Make detections
            image, results = utils.mediapipe_detection(frame, holistic)
            
            # Draw detections over the frame
            utils.draw_styled_landmarks(image, results)
            
            # 2. Prediction Logic
            #extract the keypoints from the frame
            keypoints = utils.extract_keypoints(results) 
            # appending the keypoints to the end of the sequence array
            sequence.append(keypoints) 
            
            # If we have captured 30 frames predict data
            if len(sequence) == 30:
                # Using data from sequence, use the model to predict the sign
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                # Prints string for predicted action in console
                print(actions[np.argmax(res)])
                # Clears frame data for next 30 frames
                sequence = []
                
            # 3. Visualization Logic
                # with np.argmax we find the value with the highest probability and we then verify if it's above the threshold
                # and if the prediction is not n-a or empty 
                if res[np.argmax(res)] > threshold and word != actions[np.argmax(res)] and actions[np.argmax(res)] != 'n-a' and actions[np.argmax(res)] != 'empty':
                    
                    # Setting word to the prediction string
                    word = (actions[np.argmax(res)])
                    
                    # Preparing to visualize prediction
                    cv2.rectangle(image, (0,0), (640,40), (245,115,16), -1)
                    cv2.putText(image, word, (3,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    # Visualize prediction overlaid on video feed
                    cv2.imshow('OpenCV Camera Feed', image)
                    # Wait two seconds to start collecting frame data again.
                    cv2.waitKey(2000)
                    word = ""
            
            # Show the camera feed
            cv2.imshow('OpenCV Camera Feed', image)
            
            # If pressing 'q' Quit the camera
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        # Close video capture
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


if __name__ == '__main__':
    main()