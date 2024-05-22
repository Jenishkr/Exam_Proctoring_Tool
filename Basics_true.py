import cv2
import face_recognition

# Load the reference image (Profile pic)
imgJk = face_recognition.load_image_file('ImagesBasic/Profile pic.jpg')
imgJk = cv2.cvtColor(imgJk, cv2.COLOR_BGR2RGB)

# Load the test image (Jenish)
imgTest = face_recognition.load_image_file('ImagesBasic/Bill gates.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# Detect and encode the face in the reference image (Profile pic)
faceLoc = face_recognition.face_locations(imgJk)[0]  # Find face location in the reference image
encodeJk = face_recognition.face_encodings(imgJk)[0]  # Encode the detected face
cv2.rectangle(imgJk, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)  # Draw a rectangle around the detected face

# Detect and encode the face in the test image (Jenish)
faceLocTest = face_recognition.face_locations(imgTest)[0]  # Find face location in the test image
encodeTest = face_recognition.face_encodings(imgTest)[0]  # Encode the detected face
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)  # Draw a rectangle around the detected face

# Compare the face in the test image with the reference image
results = face_recognition.compare_faces([encodeJk], encodeTest)  # Compare faces
faceDis = face_recognition.face_distance([encodeJk], encodeTest)  # Calculate the face distance
print(results, faceDis)  # Print the results and face distance

# Display the results on the test image
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

# Display the reference image (Profile pic) and the test image (Jenish)
cv2.imshow('Profile pic', imgJk)
cv2.imshow('Jenish', imgTest)

cv2.waitKey(0)  # Wait for a key press to close the OpenCV windows
