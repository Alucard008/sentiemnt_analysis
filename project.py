#!/usr/bin/env python
# coding: utf-8

# In[71]:


import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tqdm import tqdm


# In[72]:



base_path = r"D:\Data Management & AI France\Deep Learning\dataset"
image_dir = os.path.join(base_path, "project_dataset")  
csv_path = os.path.join(image_dir, "labels.csv")


df = pd.read_csv(csv_path).drop(columns=['Unnamed: 0'], errors='ignore')
df.info()


# In[73]:


df.head()


# In[74]:


df.isnull().sum()


# In[75]:


label_counts = df['label'].value_counts()

plt.figure(figsize=(10, 6))
label_counts.plot(kind='bar')
plt.xlabel("Emotion Labels")
plt.ylabel("Number of Images")
plt.title("Distribution of Emotion Labels in Dataset")
plt.xticks(rotation=45)
plt.show()

print("Label Distribution:\n", label_counts)


# In[76]:


valid_images = []
valid_labels = []


# In[77]:


def preprocess_image(image_path, target_size=(128, 128)):
    if not os.path.exists(image_path):
        return None
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        image = cv2.resize(image, target_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        return image
    except Exception:
        return None

for index, row in tqdm(df.iterrows(), total=len(df)):
    image_path = os.path.join(image_dir, row["pth"])
    processed_image = preprocess_image(image_path)
    if processed_image is not None:
        valid_images.append(processed_image)
        valid_labels.append(row["label"])


# In[78]:


if len(valid_images) > 0:
    X = np.array(valid_images, dtype=np.float32)  # Uses 4x less memory
    y = np.array(valid_labels)
    
    np.save(os.path.join(base_path, "X_preprocessed.npy"), X)
    np.save(os.path.join(base_path, "y_labels.npy"), y)

    print(f"Total valid images: {len(X)}")
    print(f"Unique labels: {np.unique(y)}")

    
    summary_df = pd.DataFrame({"Labels": y})
    summary_df
else:
    print("No valid images found. Check data paths and preprocessing.")


# In[80]:


X = np.array(valid_images, dtype=np.float32)
y = np.array(valid_labels)


# In[ ]:


np.save(os.path.join(base_path, "X_preprocessed.npy"), X)
np.save(os.path.join(base_path, "y_labels.npy"), y)


# In[ ]:


# Convert labels to categorical
unique_labels = np.unique(y)
label_map = {i: label for i, label in enumerate(unique_labels)}
reverse_label_map = {label: i for i, label in enumerate(unique_labels)}
y = np.array([reverse_label_map[label] for label in y])
y = to_categorical(y, num_classes=len(unique_labels))


# In[ ]:


# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# In[ ]:


# Define CNN model
cnn_model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(unique_labels), activation='softmax')
])


# In[ ]:


# Compile and train CNN model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
cnn_model.save(os.path.join(base_path, "cnn_model.h5"))


# In[ ]:


test_loss, test_acc = cnn_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")


# In[ ]:


X_reshaped = X.reshape(X.shape[0], -1)  # e.g., (22540, 49152)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)


# Define LSTM model for sequence-to-sequence prediction
lstm_model = keras.Sequential([
    layers.Reshape((8, X_reshaped.shape[1] // 8)),
    layers.LSTM(64, return_sequences=True),
    layers.TimeDistributed(layers.Dense(64, activation='relu')),
    layers.TimeDistributed(layers.Dense(8, activation='softmax'))  
])

# Compile the model
lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


lstm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
lstm_model.save(os.path.join(base_path, "lstm_model.h5"))


# In[ ]:


# Evaluate LSTM model
test_loss_lstm, test_acc_lstm = lstm_model.evaluate(X_test, y_test)


# In[ ]:


# Load trained models for prediction
def preprocess_image_for_cnn(image_path):
    print("image path -------------------------------------------------------" , image_path)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_image_for_lstm(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img = img.flatten().reshape(1, -1)
    return img


# In[ ]:





cnn_model = keras.models.load_model(os.path.join(base_path, "cnn_model.h5"))
lstm_model = keras.models.load_model(os.path.join(base_path, "lstm_model.h5"))

# CNN Prediction
processed_img_cnn = preprocess_image_for_cnn("D:/Data Management & AI France/Deep Learning/dataset/project_dataset/sad/image0001948.jpg")
predictions_cnn = cnn_model.predict(processed_img_cnn)
print("prdic" , predictions_cnn)
predicted_label_cnn = label_map.get(np.argmax(predictions_cnn), "Unknown")
# LSTM Prediction
processed_img_lstm = preprocess_image_for_lstm("D:/Data Management & AI France/Deep Learning/dataset/project_dataset/sad/image0001948.jpg")
predictions_lstm = lstm_model.predict(processed_img_lstm)
predicted_label_lstm = label_map.get(np.argmax(predictions_lstm), "Unknown")

# Compare Model Performance
# print(f"CNN Test Accuracy: {test_acc_cnn:.4f}")
print(f"LSTM Test Accuracy: {test_acc_lstm:.4f}")
print(f"Predicted Sentiment (CNN): {predicted_label_cnn}")
print(f"Predicted Sentiment (LSTM): {predicted_label_lstm}")



# In[ ]:




# In[ ]:


# # Reshape X into sequences for LSTM
# # Here, X is reshaped from (num_samples, height, width, channels) to (num_samples, height*width*channels)
# X_reshaped = X.reshape(X.shape[0], -1)  # e.g., (22540, 49152)

# # Split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# # If y is already one-hot encoded to shape (22540, 8, 2), do not one-hot encode again.
# # If not, you would need to convert y into that shape.
# # unique_labels = np.unique(y)  # Only use this if y is integer labels.
# # y_train = to_categorical(y_train, num_classes=len(unique_labels))
# # y_test = to_categorical(y_test, num_classes=len(unique_labels))

# # Define LSTM model for sequence-to-sequence prediction
# lstm_model = keras.Sequential([
#     # Reshape flattened input into sequences of length 8,
#     # where each timestep has 49152 // 8 = 6144 features.
#     layers.Reshape((8, X_reshaped.shape[1] // 8)),
    
#     # LSTM outputs a sequence of 8 timesteps, each with 64 features.
#     layers.LSTM(64, return_sequences=True),
    
#     # Apply Dense layers at each timestep.
#     layers.TimeDistributed(layers.Dense(64, activation='relu')),
#     layers.TimeDistributed(layers.Dense(2, activation='softmax'))  # 2 classes per timestep.
# ])

# # Compile the model
# lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# In[ ]:


# print("X_train shape:", X_train.shape)  # (samples, 150528)
# print("y_train shape:", y_train.shape)  # (samples, 2)


# In[ ]:


# # Train the model
# lstm_history = lstm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)


# In[ ]:


# # Save LSTM model
# lstm_model.save(os.path.join(base_path, "lstm_model.h5"))


# In[ ]:


# # Evaluate LSTM model
# test_loss_lstm, test_acc_lstm = lstm_model.evaluate(X_test, y_test)
# print(f"LSTM Test Accuracy: {test_acc_lstm:.4f}")


# In[ ]:


# # Compare Model Performance
# print(f"Comparison: CNN Accuracy = {test_acc_cnn:.4f}, LSTM Accuracy = {test_acc_lstm:.4f}")


# In[ ]:


# # Load trained LSTM model
# lstm_model_path = os.path.join(base_path, "lstm_model.h5")
# lstm_model = keras.models.load_model(lstm_model_path)

# # Load label mapping
# y = np.load(os.path.join(base_path, "y_labels.npy"), allow_pickle=True)
# unique_labels = np.unique(y)
# label_map = {idx: label for idx, label in enumerate(unique_labels)}

# # Function to preprocess an image for LSTM

# def preprocess_image_for_lstm(image_path):
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (128, 128))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # Normalize
#     img = img.flatten().reshape(1, -1)  # Reshape for LSTM model
#     return img

# # Provide a test image path
# test_image_path = os.path.join(base_path, "D:/Data Management & AI France/Deep Learning/dataset/project_dataset/sad/image0012446.jpg")  # Change to an actual test image

# # Preprocess and predict
# processed_img = preprocess_image_for_lstm(test_image_path)
# predictions = lstm_model.predict(processed_img)
# predicted_label = label_map[np.argmax(predictions)]

# print(f"Predicted Sentiment (LSTM): {predicted_label}")


# In[ ]:





# In[ ]:





# In[ ]:


# ## Test the Model

# # Load trained model
# model_path = os.path.join(base_path, "cnn_model.h5")
# model = keras.models.load_model(model_path)

# # Load label mapping
# y = np.load(os.path.join(base_path, "y_labels.npy"), allow_pickle=True)
# unique_labels = np.unique(y)
# label_map = {idx: label for idx, label in enumerate(unique_labels)}

# # Function to preprocess an image
# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (128, 128))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # Normalize
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
#     return img

# # Provide a test image path
# test_image_path = os.path.join(base_path, "D:/Data Management & AI France/Deep Learning/dataset/project_dataset/sad/image0012446.jpg")  # Change to an actual test image

# # Preprocess and predict
# processed_img = preprocess_image(test_image_path)
# predictions = model.predict(processed_img)
# predicted_label = label_map[np.argmax(predictions)]

# print(f"Predicted Sentiment: {predicted_label}")

