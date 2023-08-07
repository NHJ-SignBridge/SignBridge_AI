import pickle
import random
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize the webcam, if not working properly, try changing 
# the constant to 0, 1, or 2
# 웹캠을 실행시킵니다. 만약 열리지 않는다면 숫자를 0, 1, 혹은 2로 바꿔주세요
cap = cv2.VideoCapture(1)

def get_random_alphabet():
    choices = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return random.choice(choices)

show_start_message = True
show_start_message_time = 0

current_alphabet = get_random_alphabet()

correct_answer_displayed = False
correct_answer_time = 0
next_question_time = 0


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 
               8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Initialize constants
skip_button_width = 100
skip_button_height = 50

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    # Calculate skip button coordinates for top-right corner
    frame_height, frame_width, _ = frame.shape
    skip_button_x = frame_width - skip_button_width
    skip_button_y = 0
    
    # Fill the skip button area with white background
    button_background = (255, 255, 255)  # White color in BGR
    frame[skip_button_y:skip_button_y + skip_button_height, skip_button_x:skip_button_x + skip_button_width] = button_background
    
    # Draw the border of the skip button area
    cv2.rectangle(frame, (skip_button_x, skip_button_y), (skip_button_x + skip_button_width, skip_button_y + skip_button_height), (0, 0, 0), 2)
    
    # Add "Skip" text inside the skip button area
    text = "Skip"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = skip_button_x + (skip_button_width - text_size[0]) // 2
    text_y = skip_button_y + (skip_button_height + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), font, 1, (0, 0, 0), 2, cv2.LINE_AA)


    if show_start_message:
            text = "Show your hand to start the quiz"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (255, 255, 51) # Note that opencv uses B G R colour code
            thickness = 2
            
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = int((frame.shape[1] - text_size[0]) / 2)
            text_y = int((frame.shape[0] + text_size[1]) / 2)

            cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, thickness)
        

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
    
    

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10
        
        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]


        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
        
        

        # Check if the user made the correct gesture
        # 사용자가 맞는 수어를 보였는지 확인합니다
        if predicted_character == current_alphabet and not correct_answer_displayed:
            show_start_message = False
            correct_answer_displayed = True
            correct_answer_time = time.time()

        if correct_answer_displayed and time.time() - correct_answer_time >= 3:
            correct_answer_displayed = False
            next_question_time = time.time()
            current_alphabet = get_random_alphabet()

        # Display "Correct!" message for 3 seconds
        # 정답 시 정답이라는 문구를 3초간 보여줍니다
        if correct_answer_displayed:
            text = "Correct!"
            font = cv2.FONT_HERSHEY_SIMPLEX
            position = (100, 100)
            font_scale = 1
            font_color = (255, 255, 51)
            thickness = 2
            cv2.putText(frame, text, position, font, font_scale, font_color, thickness)

         # Check if the "Skip" button is pressed to show the next question
        if skip_button_x <= x <= skip_button_x + skip_button_width and skip_button_y <= y <= skip_button_y + skip_button_height:
            current_alphabet = get_random_alphabet()

        # Check if it's time to show the next question (either through correct answer or skip button)
        if predicted_character == current_alphabet and not correct_answer_displayed or time.time() - next_question_time >= 3:
            correct_answer_displayed = False
            next_question_time = time.time()
            current_alphabet = get_random_alphabet()
        
        text = "Sign " + current_alphabet + "."
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (50, 50)
        font_scale = 1
        font_color = (255, 255, 51)
        thickness = 2
        cv2.putText(frame, text, position, font, font_scale, font_color, thickness)


        # Display the frame with overlaid text and hand landmarks
        cv2.imshow("Webcam", frame)

        # Break the loop if the 'q' key is pressed
        # q를 누를 시 프로그램을 종료합니다
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

    # clear the data for the next iteration
    # 데이터를 클리어 해줍니다
    data_aux.clear()
    x_.clear()
    y_.clear()


cap.release()
cv2.destroyAllWindows()
