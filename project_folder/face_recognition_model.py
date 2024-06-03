import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from mtcnn import MTCNN
from keras_facenet import FaceNet
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import load_img, img_to_array

DATA_DIR = '/Users/m/Desktop/project_folder/face/lfw'  # actual path to the dataset
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

def load_data(data_dir, image_size):
    X = []
    y = []
    detector = MTCNN()
    for person_folder in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_folder)
        if os.path.isdir(person_dir):
            images = os.listdir(person_dir)
            for image_file in images:
                img = load_img(os.path.join(person_dir, image_file), target_size=image_size)
                img_array = img_to_array(img)
                X.append(img_array)
                y.append(person_folder)
    X = np.array(X)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)  # Encode labels as integers
    
    # Save label encoder classes
    np.save('label_encoder_classes.npy', label_encoder.classes_)
    
    return X, y, label_encoder  # Return integer-encoded labels and label encoder
# Load and preprocess data
X, y, label_encoder = load_data(DATA_DIR, IMAGE_SIZE)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize FaceNet model
facenet = FaceNet()

# Extract face embeddings
X_train_embeddings = facenet.embeddings(X_train)
X_val_embeddings = facenet.embeddings(X_val)

num_classes = len(label_encoder.classes_)  # Calculate the number of unique classes in your dataset

# Define classification model using the functional API
inputs = layers.Input(shape=(X_train_embeddings.shape[1],))  # Input layer with the shape of embeddings
x = layers.Dense(128, activation='relu')(inputs)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
checkpoint_cb = callbacks.ModelCheckpoint('face_recognition_model.h5', save_best_only=True)
history = model.fit(X_train_embeddings, y_train, validation_data=(X_val_embeddings, y_val),
                    batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[checkpoint_cb])

# Save the label encoder classes
np.save('label_encoder_classes.npy', label_encoder.classes_)

# Evaluate the model
model.load_weights('face_recognition_model.h5')
loss, accuracy = model.evaluate(X_val_embeddings, y_val)
print(f'Validation loss: {loss:.4f}, Validation accuracy: {accuracy:.4f}')

