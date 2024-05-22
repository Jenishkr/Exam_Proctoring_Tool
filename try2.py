import cv2
import face_recognition

# Load the reference image (Elon Musk)
imgElon = face_recognition.load_image_file('ImagesBasic/Jenish.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
encodeElon = face_recognition.face_encodings(imgElon)[0]  # Encode the reference image

# Access the webcam feed
cap = cv2.VideoCapture(0)  # 0 represents the default camera (usually the built-in webcam)

frame_count = 0
match_count = 0

while True:
    ret, frame = cap.read()  # Read a frame from the webcam

    # Detect face location and encode the frame from the webcam
    faceLocTest = face_recognition.face_locations(frame)
    if len(faceLocTest) > 0:
        encodeTest = face_recognition.face_encodings(frame, faceLocTest)[0]
        cv2.rectangle(frame, (faceLocTest[0][3], faceLocTest[0][0]), (faceLocTest[0][1], faceLocTest[0][2]), (255, 0, 255), 2)

        # Compare the captured frame with the reference image
        results = face_recognition.compare_faces([encodeElon], encodeTest)
        faceDis = face_recognition.face_distance([encodeElon], encodeTest)
        cv2.putText(frame, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        frame_count += 1
        if results[0]:
            match_count += 1

    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Calculate and display accuracy percentage
if frame_count > 0:
    accuracy_percentage = (match_count / frame_count) * 100
    print(f'Accuracy: {accuracy_percentage:.2f}%')

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()