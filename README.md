# Face-Recognition-Attendance
Face recognition attendance project using OpenCV Python. We'll use a module called face recognition to build encodings of 128 values that are unique to each face.

We'll start by providing the location to a folder containing the photographs of the persons that need to be recognised. 
The face encodings method in the face recognition module will then return encodings for each of the faces in the folder, which will be stored in a list.
And When a face is displayed on the camera, the encodings for that face are generated and compared to the available encodings to map the face. 
The name (picture name) and the time will be recorded on the sheet once the face has been recognised.
