from keras.utils import to_categorical
from keras.models import Sequential
from keras.preprocessing.image import load_img
# from tensorflow.keras.utils import load_img
# from tensorflow.keras.preprocessing.image import load_img
# from keras_preprocessing.image import load_img
# from tensorflow.keras.utils import load_img
import keras
from tqdm.notebook import tqdm
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

TRAIN_DIR = 'images/train'
TEST_DIR = 'images/test'

def createdataset(dir):
    img_paths = []
    labels = []
    for label in os.listdir(dir):
        for img in os.listdir(os.path.join(dir, label)):
            img_paths.append(os.path.join(dir, label, img))
            labels.append(label)
        print(label,"completed")
    return img_paths,labels

# createdeatset(TRAIN_DIR)

train = pd.DataFrame()
train['image'],train['label'] = createdataset(TRAIN_DIR)

test = pd.DataFrame()
test['image'], test['label'] = createdataset(TEST_DIR)

# def extract_features(images):
#     features = []
#     for image in tqdm(images):
#         img = load_img(image,grayscale =  True )
#         img = np.array(img)
#         features.append(img)
#     features = np.array(features)
#     features = features.reshape(len(features),48,48,1)
#     return features

def extract_features(images):
    features = []
    total_images = len(images)
    
    for i, image in enumerate(images):
        print(f"Processing image {i+1}/{total_images}")
        img = load_img(image, grayscale=True)
        img = np.array(img)
        features.append(img)
        
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features


train_features = extract_features(train['image']) 
test_features = extract_features(test['image'])
x_train = train_features/255.0
x_test = test_features/255.0


le = LabelEncoder()
le.fit(train['label'])
y_train = le.transform(train['label'])
y_test = le.transform(test['label'])
y_train = to_categorical(y_train,num_classes = 7)
y_test = to_categorical(y_test,num_classes = 7)
model = Sequential()
# convolutional layers
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
# fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(7, activation='softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = 'accuracy' )
model.fit(x= x_train,y = y_train, batch_size = 128, epochs = 100, validation_data = (x_test,y_test)) 
model_json = model.to_json()
with open("emotiondetector.json",'w') as json_file:
    json_file.write(model_json)
model.save("emotiondetector.h5")
from keras.models import model_from_json
json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("facialemotionmodel.h5")
label = ['angry','disgust','fear','happy','neutral','sad','surprise']
def ef(image):
    img = load_img(image,grayscale =  True )
    feature = np.array(img)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0
    
