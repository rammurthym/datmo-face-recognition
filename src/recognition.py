import face_recognition
import os
import pandas as pd
import numpy as np
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from scikit_checkpoint import ScikitCheckpoint
from sklearn.cluster import KMeans
np.random.seed(5)

data = []
target = []
face_names = []
num_faces = len(face_names)
faces_selected = set()

for file_path in glob(os.environ.get("INPUT_DIR")+"/*/*"):
    class_name = file_path.split('/')[-2]
    if not class_name == 'test':
        if class_name not in faces_selected:
            faces_selected.add(class_name)
            face_names.append(class_name)
        load_image = face_recognition.load_image_file(file_path)
        list_encoding = face_recognition.face_encodings(load_image)
        if len(list_encoding) > 0:
            face_encoding = list_encoding[0]
            data.append(face_encoding)
            target.append(face_names.index(class_name))

face_names = np.array(face_names)
data = np.asarray(data)
dimensions = range(len(face_encoding))

df = pd.DataFrame(data, columns=dimensions)
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .60
df['face'] = pd.Categorical.from_codes(target, face_names)
df.head()

train, test = df[df['is_train']==True], df[df['is_train']==False]

features = df.columns[:len(face_encoding)]
clf = RandomForestClassifier(n_jobs=4)
y, _ = pd.factorize(train['face'])
clf.fit(train[features], y)
checkpoint = ScikitCheckpoint(os.environ['SNAPSHOTS_DIR'], )
stats = {'label': 'random_forest'}

checkpoint.save_model(clf, stats)
preds = face_names[np.array(clf.predict(test[features]))]
preds = face_names[clf.predict(test[features])]
cross_validation = pd.crosstab(test['face'], preds, rownames=['actual'], colnames=['preds'])
print(cross_validation)