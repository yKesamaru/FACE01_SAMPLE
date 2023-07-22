"""License for the Code.

Copyright Owner: Yoshitsugu Kesamaru
Please refer to the separate license file for the license of the code.
"""

"""Example of calculating similarity from two photos with efficientnetv2_arcface.onnx model.

Summary:
    An example of loading face images with file names listed in 
    average_face.txt and creating an average face from those images.

Example:
    .. code-block:: bash
    
        python3 example/similarity.py
        
Source code:
    `similarity.py <../example/similarity.py>`_
"""
# Operate directory: Common to all examples
import cv2
import numpy as np
import os
import glob
import mediapipe as mp

# Initializing the Mediapipe face landmark detector
mp_face_mesh = mp.solutions.face_mesh  # type: ignore
face_mesh = mp_face_mesh.FaceMesh()

# Change directory where average_face.txt exists
# root_dir = '/home/terms/bin/FACE01'
# os.chdir(root_dir)

def align_face(image):
    # Detecting face landmarks
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Calculating the center of the face
            center = np.mean([[data.x, data.y] for data in face_landmarks.landmark], axis=0).astype("int")

            # Calculating the angle of the face
            dX = face_landmarks.landmark[33].x - face_landmarks.landmark[263].x
            dY = face_landmarks.landmark[33].y - face_landmarks.landmark[263].y
            angle = np.degrees(np.arctan2(dY, dX)) - 180

            # Calculating the center of the image
            (h, w) = image.shape[:2]
            (cX, cY) = (w // 2, h // 2)

            # Rotating the image to align the face frontally
            M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
            aligned = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            return aligned

# Loading the directory names
with open(os.path.join('example', 'average_face.txt'), 'r') as f:
    male_dirs = f.read().splitlines()


# Reading images, aligning faces, and resizing
images = []
for dir_name in male_dirs:
# for dir_name in male_dirs + female_dirs:
    image_files = glob.glob(f'{dir_name}/*.png')
    for image_file in image_files:
        image = cv2.imread(image_file)
        aligned = align_face(image)
        resized = cv2.resize(aligned, (224, 224))  # type: ignore
        images.append(resized)

# Converting images to a numpy array
images = np.array(images)

# Calculating the average face
average_face = np.mean(images, axis=0).astype("uint8")

# Displaying the average face
cv2.imshow("Average Face", average_face)
cv2.waitKey(0)

