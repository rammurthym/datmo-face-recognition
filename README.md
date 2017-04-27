## Face-Recognition

  This repository comprises main tasks required to be done using facial recognition

  ### 1. Face Detection (detection.py)
        a. To find bounding box for faces
        b. To find number of people in any image
        Usage: Application can range like Facebook using this to find count of people and detect face to let users tag
         it
  ### 2. Facial Landmark Detection (landmark_detection.py)
        a. To find the face feature locations (eyes, nose, etc) for each face in the image
            Usage: Application can lie to apply makeup application or is the founding basis to
             overlay structure as Snapchat does for lenses (https://support.snapchat.com/en-US/article/lenses1)
  ### 3. Facial Verification (verification.py) <br />
        a. Given an image of face, we can compare if any new image contains the same person.
            Usage: Widely used application eg. Uber uses to check if driver is the same person as registered on app.
            (https://newsroom.uber.com/securityselfies/)
  ### 4. Facial Recognition (recognition.py)
        a. In a given number of classes does any new image of a face lie in one of the classes.
            Usage: Application lie to identify any celebrety and also used by Facebook to identify friends in
             moments or by Google in their Photos app.


  Built using dlib and face_recognition by Adam (https://github.com/ageitgey/face_recognition) 