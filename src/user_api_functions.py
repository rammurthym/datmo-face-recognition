import urllib, cStringIO
import scipy 
import pickle
import os
import numpy as np
import face_recognition

filename = os.path.join(os.environ['INPUT_DIR'],'model.dat')
clf = pickle.load( open(filename , "rb" ) )

def add(params):
    return params['a'] + params['b']


def recognition(params):
    """
    Loads an image url (.jpg, .png, etc) into a numpy array
    :param url: image url to load
    :return: face recognition over image url
    """
    image_file = cStringIO.StringIO(urllib.urlopen(params['url']).read())
    image = scipy.misc.imread(image_file, mode='RGB')
    # read the image file in a numpy array
    list_encoding = face_recognition.face_encodings(image)
    test_pred = []
    face_names = np.array(['other', 'donald_trump'])
    test_preds = []
    if list_encoding:
        for encoding in list_encoding:
            proba = clf.predict_proba([encoding]).reshape(1, -1)
            proba = proba.reshape(1, -1)
            if proba[0][1] >= 0.7:
                test_pred = face_names[1]
                test_preds.append(test_pred)                
            else:
                test_pred = face_names[0]
                test_preds.append(test_pred)                
    return test_preds
