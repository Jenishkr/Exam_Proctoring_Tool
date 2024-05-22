import cv2
import face_recognition

# Load the reference image (Elon Musk)
imgElon = face_recognition.load_image_file('ImagesBasic/Elon Musk.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

# Load the test image (Bill Gates)
imgTest = face_recognition.load_image_file('ImagesBasic/Bill Gates.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# Detect and encode the face in the reference image (Elon Musk)
faceLoc = face_recognition.face_locations(imgElon)[0]  # Find face location in the reference image
encodeElon = face_recognition.face_encodings(imgElon)[0]  # Encode the detected face
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)  # Draw a rectangle around the detected face

# Detect and encode the face in the test image (Bill Gates)
faceLocTest = face_recognition.face_locations(imgTest)[0]  # Find face location in the test image
encodeTest = face_recognition.face_encodings(imgTest)[0]  # Encode the detected face
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)  # Draw a rectangle around the detected face

# Compare the face in the test image with the reference image
results = face_recognition.compare_faces([encodeElon], encodeTest)  # Compare faces
faceDis = face_recognition.face_distance([encodeElon], encodeTest)  # Calculate the face distance
print(results, faceDis)  # Print the results and face distance

# Display the results on the test image
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

# Display the reference image (Elon Musk) and the test image (Bill Gates)
cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Bill Gates Test', imgTest)

cv2.waitKey(0)  # Wait for a key press to close the OpenCV windows
