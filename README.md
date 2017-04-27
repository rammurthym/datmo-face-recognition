### Face-Recognition

  This repository comprises main tasks required to be done using facial recognition

  # 1. Face Detection (detection.py) <br />
        a. To find bounding box for faces <br />
        b. To find number of people in any image <br />
        Usage: Application can range like Facebook using this to find count of people and detect face to let users tag
         it <br />
  # 2. Facial Landmark Detection (landmark_detection.py) <br />
        a. To find the face feature locations (eyes, nose, etc) for each face in the image <br />
            Usage: Application can lie to apply makeup application or is the founding basis to <br />
             overlay structure as Snapchat does for lenses (https://support.snapchat.com/en-US/article/lenses1) <br />
  # 3. Facial Verification (verification.py) <br />
        a. Given an image of face, we can compare if any new image contains the same person. <br />
            Usage: Widely used application eg. Uber uses to check if driver is the same person as registered on app. <br />
            (https://newsroom.uber.com/securityselfies/) <br />
  # 4. Facial Recognition (recognition.py) <br />
        a. In a given number of classes does any new image of a face lie in one of the classes. <br />
            Usage: Application lie to identify any celebrety and also used by Facebook to identify friends in
             moments or by Google in their Photos app. <br />


  Built using dlib and face_recognition by Adam (https://github.com/ageitgey/face_recognition) <br />
