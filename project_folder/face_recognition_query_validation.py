import os
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from keras_facenet import FaceNet
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report

# Define constants
BASE_DIR = '/Users/m/Desktop/project_folder'
DATA_DIR = os.path.join(BASE_DIR, 'face/lfw')  # Datapath
IMAGE_SIZE = (224, 224)

# Configure logging
logging.basicConfig(filename=os.path.join(BASE_DIR, 'face_recognition.log'), level=logging.ERROR)

# Load model and label encoder
model_path = os.path.join(BASE_DIR, 'face_recognition_model.h5')
label_encoder_path = os.path.join(BASE_DIR, 'label_encoder_classes.npy')
try:
    model = load_model(model_path)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(label_encoder_path, allow_pickle=True)
except Exception as e:
    logging.error(f"Error loading model or label encoder: {e}")
    raise  # Raising exception to stop execution if loading fails

def preprocess_image(image_path, image_size):
    try:
        img = load_img(image_path, target_size=image_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logging.error(f"Error loading image: {image_path}\n{e}")
        return None  # Return None or handle the error as needed

def extract_embeddings(facenet, img_array):
    embeddings = facenet.embeddings(img_array)
    return embeddings

# Load the FaceNet model for embedding extraction
facenet = FaceNet()

# Define query and validation image directories
query_dir = os.path.join(BASE_DIR, 'query_images')  # Directory containing query images
validation_dir = os.path.join(BASE_DIR, 'validation_images')  # Directory containing validation images

# Get list of all person folders in the query directory
query_person_folders = [folder for folder in os.listdir(query_dir) if os.path.isdir(os.path.join(query_dir, folder))]

# Preprocess and extract embeddings for query images
query_embeddings = []
query_true_labels = []
for person_folder in query_person_folders:
    person_dir = os.path.join(query_dir, person_folder)
    images = os.listdir(person_dir)
    for image_file in images:
        img_path = os.path.join(person_dir, image_file)
        img_array = preprocess_image(img_path, IMAGE_SIZE)
        if img_array is not None:
            embeddings = extract_embeddings(facenet, img_array)
            query_embeddings.append(embeddings)
            query_true_labels.append(person_folder)

if len(query_embeddings) > 0:
    query_embeddings = np.array(query_embeddings).reshape(len(query_embeddings), -1)

# Get list of all person folders in the validation directory
validation_person_folders = [folder for folder in os.listdir(validation_dir) if os.path.isdir(os.path.join(validation_dir, folder))]

# Preprocess and extract embeddings for validation images
validation_embeddings = []
validation_true_labels = []
for person_folder in validation_person_folders:
    person_dir = os.path.join(validation_dir, person_folder)
    images = os.listdir(person_dir)
    for image_file in images:
        img_path = os.path.join(person_dir, image_file)
        img_array = preprocess_image(img_path, IMAGE_SIZE)
        if img_array is not None:
            embeddings = extract_embeddings(facenet, img_array)
            validation_embeddings.append(embeddings)
            validation_true_labels.append(person_folder)

if len(validation_embeddings) > 0:
    validation_embeddings = np.array(validation_embeddings).reshape(len(validation_embeddings), -1)

# Perform face recognition on query images
if len(query_embeddings) > 0:
    query_probabilities = model.predict(query_embeddings)
    query_predictions = np.argmax(query_probabilities, axis=1)
    query_predicted_labels = label_encoder.inverse_transform(query_predictions)

# Perform face recognition on validation images
if len(validation_embeddings) > 0:
    validation_probabilities = model.predict(validation_embeddings)
    validation_predictions = np.argmax(validation_probabilities, axis=1)
    validation_predicted_labels = label_encoder.inverse_transform(validation_predictions)

# Calculate performance metrics and save reports if data is available
if len(query_embeddings) > 0 and len(query_true_labels) > 0:
    query_report = classification_report(query_true_labels, query_predicted_labels, zero_division=1)
    with open(os.path.join(BASE_DIR, 'query_classification_report.txt'), 'w') as f:
        f.write(query_report)
    print("Query Images Recognition Report saved.")

if len(validation_embeddings) > 0 and len(validation_true_labels) > 0:
    validation_report = classification_report(validation_true_labels, validation_predicted_labels, zero_division=1)
    with open(os.path.join(BASE_DIR, 'validation_classification_report.txt'), 'w') as f:
        f.write(validation_report)
    print("Validation Images Recognition Report saved.")
else:
    print("No data available for evaluation.")
