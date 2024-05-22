import cv2
import face_recognition

# Load the reference image (Jenish)
imgJenish = face_recognition.load_image_file('ImagesBasic/Jenish.jpg')  # Load the reference image
imgJenish = cv2.cvtColor(imgJenish, cv2.COLOR_BGR2RGB)  # Convert the image to RGB format
encodeJenish = face_recognition.face_encodings(imgJenish)[0]  # Encode the reference image

# Access the webcam feed
cap = cv2.VideoCapture(0)  # 0 represents the default camera (usually the built-in webcam)

image_counter = 1  # Initialize the image counter

while True:
    ret, frame = cap.read()  # Read a frame from the webcam

    # Detect face location and encode the frame from the webcam
    faceLocTest = face_recognition.face_locations(frame)  # Detect face locations in the frame
    if len(faceLocTest) > 0:
        encodeTest = face_recognition.face_encodings(frame, faceLocTest)[0]  # Encode the detected face
        cv2.rectangle(frame, (faceLocTest[0][3], faceLocTest[0][0]), (faceLocTest[0][1], faceLocTest[0][2]), (255, 0, 255), 2)

        # Compare the captured frame with the reference image
        results = face_recognition.compare_faces([encodeJenish], encodeTest)  # Compare faces
        faceDis = face_recognition.face_distance([encodeJenish], encodeTest)  # Calculate the face distance
        cv2.putText(frame, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        key = cv2.waitKey(1)  # Wait for a key press
        if key & 0xFF == ord('q'):  # Press 'q' to exit the loop
            break
        elif key & 0xFF == ord('c'):  # Press 'c' to capture an image
            image_filename = f'{image_counter}.jpg'
            cv2.imwrite(image_filename, frame)  # Save the captured image with the image_counter as the filename
            print(f"Image {image_counter} captured and saved as {image_filename}.")
            image_counter += 1  # Increment the image counter

    cv2.imshow('Webcam Feed', frame)  # Display the frame with face recognition

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
