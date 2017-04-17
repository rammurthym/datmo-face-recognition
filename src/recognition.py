import face_recognition
import os
import pandas as pd
import numpy as np
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from scikit_checkpoint import ScikitCheckpoint


data = []
target = []
face_names = ['donald_trump', 'mike_pence', 'putin']
i = 0
for file_path in glob.glob(os.environ.get("INPUT_DIR")+"/*/*"):
    print(i)
    i+=1
    load_image = face_recognition.load_image_file(file_path)
    face_encoding = face_recognition.face_encodings(load_image)[0]
    data.append(face_encoding)
    class_name = file_path.split('/')[-1]
    target.append(face_names.index(class_name))


data = np.asarray(data)
dimensions = range(len(face_encoding))

df = pd.DataFrame(data, columns=face_encoding)
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
df['face'] = pd.Categorical.from_codes(target, face_names)
df.head()

train, test = df[df['is_train']==True], df[df['is_train']==False]

features = df.columns[:len(face_encoding)]
clf = RandomForestClassifier(n_jobs=4)
y, _ = pd.factorize(train['species'])
clf.fit(train[features], y)
checkpoint = ScikitCheckpoint(os.environ['SNAPSHOTS_DIR'], )
stats = {'label': 'random_forest'}
checkpoint.save_model(clf, stats)
preds = face_names[clf.predict(test[features])]
pd.crosstab(test['face'], preds, rownames=['actual'], colnames=['preds'])