import urllib, cStringIO
import scipy 
import pickle
import os
import numpy as np
import face_recognition
import json


filename = os.path.join(os.environ['INPUT_DIR'],'model.dat')
clf = pickle.load( open(filename , "rb" ) )
filename = os.path.join(os.environ['SHARED_OUTPUT_DIR'],'face_names.pkl')
face_names = np.array(pickle.load(open(filename, 'rb')))

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
    test_preds = []
    if list_encoding:
        for encoding in list_encoding:
            test_pred = face_names[clf.predict([encoding])][0]
            test_preds.append(test_pred)
    return list(test_preds)
