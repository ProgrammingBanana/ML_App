import cv2
import numpy as np
from mp_utils import mp_utilities
from tensorflow import keras


def main():
    utils = mp_utilities()
    actions = np.array(['ayuda', 'clase', 'donde', 'gracias', 'hola', 'necesitar', 'no_entender', 'repetir', 'n-a', 'empty'])
    model = keras.models.load_model("actions_models/actions_3.h5")
    sequence = [] # Append data over 30 frames
    word = "" # Allows us to concatenate list of predictions
    threshold = 0.45 # Confidence metric (only render results if it meets our threshold value)

    # Connects to the webcam (the number can vary depending on machine)
    cap = cv2.VideoCapture(0)

    # Set mediapipe model
    with utils.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            
            # Read feed
            ret, frame = cap.read()
            
            # Make detections
            image, results = utils.mediapipe_detection(frame, holistic)
            
            utils.draw_styled_landmarks(image, results)
            
            # 2. Prediction Logic
            keypoints = utils.extract_keypoints(results) #extract the keypoints from the frame
            sequence.append(keypoints) # appending the keypoints to the end of the sequence array
            # sequence = sequence[-30:] # grabs the last 30 frames
            
            # If we have captured 30 frames predict data
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                sequence = []
    #             print(actions[np.argmax(res)])
                
            # 3. Visualization Logic
            # with np.argmax we find the value with the highest probability and we then verify if it's above the threshold
                if res[np.argmax(res)] > threshold and word != actions[np.argmax(res)] and actions[np.argmax(res)] != 'n-a' and actions[np.argmax(res)] != 'empty':
                    
                    word = (actions[np.argmax(res)])
                    
                    cv2.rectangle(image, (0,0), (640,40), (245,115,16), -1)
                    cv2.putText(image, word, (3,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Camera Feed', image)
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