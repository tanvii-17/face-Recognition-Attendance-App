#pip install cmake
#pip install face-recognition
#pip install opencv-python
#pip inwstall numpy
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)      #0 for first webcam, 1 for second webcam ans so on...

#Load known faces
tanvi_image = face_recognition.load_image_file("faces/tanvi.jpg")
tanvi_encoding = face_recognition.face_encodings(tanvi_image)[0]

nishita_image = face_recognition.load_image_file("faces/nishita.jpg")
nishita_encoding = face_recognition.face_encodings(nishita_image)[0]

sushmita_image = face_recognition.load_image_file("faces/sushmita.jpg")
sushmita_encoding = face_recognition.face_encodings(sushmita_image)[0]

#storing the names of the encodings
known_face_encodings = [tanvi_encoding, nishita_encoding, sushmita_encoding]
known_face_names = ["Tanvi", "Nishita", "Sushmita"]

#List of expected students
students = known_face_names.copy()

face_locations = []
face_encodings = []

#Get the current date and time
now = datetime.now()                     #a function
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()          #  _ to show whether video capture was successful or not
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    #Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        #will compare known_face_encodings with the face_encoding

        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        #cosine similarity type.....will tell how much similar is the face_encoding with known_face_encodings

        best_match_index = np.argmin(face_distance)    #lesser distance means more similar face

        if(matches[best_match_index]):
            name = known_face_names[best_match_index]

            #Add the text if person is present
            if name in known_face_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 1.5
                fontColor = (255, 0, 0)
                thickness = 3
                lineType = 2

                cv2.putText(frame, name +  "Present", bottomLeftCornerOfText, font,
                            fontScale, fontColor, thickness, lineType)

                if name in students:
                    students.remove(name)

                    #Get the current time
                    current_time = now.strftime("%H:%M:%S")
                    lnwriter.writerow([name, current_time])

       #Display the frame with attendance status
        cv2.imshow("Attendance", frame)

        #Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

video_capture.release()
cv2.destroyAllWindows()
f.close()