import pickle
import random
import time 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./numbers.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly'.format(score * 100))

f = open('numbers.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

total = 0

# Initialize the webcam, if not working properly, try changing 
# the constant to 0, 1, or 2
# 웹캠을 실행시킵니다. 만약 열리지 않는다면 숫자를 0, 1, 혹은 2로 바꿔주세요
cap = cv2.VideoCapture(1)

def get_random_alphabet():
    choices = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
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

labels_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10'}

skipped = []

while True:
    data_aux_left = []  # for the left hand
    data_aux_right = []  # for the right hand
    x_ = []
    y_ = []

    ret, frame = cap.read()


    if show_start_message:
        text = "Show your hand to start the quiz"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 51)
        thickness = 2

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = int((frame.shape[1] - text_size[0]) / 2)
        text_y = int((frame.shape[0] + text_size[1]) / 2)

        cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, thickness)

    H, W, _ = frame.shape

    instructions = "Press 's' to skip the current question and 'q' to end this quiz"
    cv2.putText(frame, instructions, (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 51), 2)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Determine if it's the left or right hand
            if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < 0.5:
                data_aux = data_aux_left
            else:
                data_aux = data_aux_right

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
        
        text = "Sign " + current_alphabet + "."
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (50, 50)
        font_scale = 1
        font_color = (255, 255, 51)
        thickness = 2
        cv2.putText(frame, text, position, font, font_scale, font_color, thickness)


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
            total += 1

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

        # Display the next question after 3 seconds
        # 3초 뒤에 다음 문제로 넘어갑니다
        if time.time() - next_question_time >= 3:
            text = "Sign " + current_alphabet + "."
        
        skip_button_pressed = cv2.waitKey(1) & 0xFF == ord('s')

        if skip_button_pressed:
            skipped.append(current_alphabet)
            current_alphabet = get_random_alphabet()
            total += 1

        # Display the frame with overlaid text and hand landmarks
        # cv2.imshow("Webcam", frame)

        # Break the loop if the 'q' key is pressed
        # q를 누를 시 프로그램을 종료하고 정답률을 보고합니다
        if cv2.waitKey(1) & 0xFF == ord('q'):
            correct = total - len(skipped)
            percentage = (correct / total) * 100
            print("Correctness: " + percentage + "%")
            print("Skipped Alphabets: ", skipped)
            break
        
    
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

