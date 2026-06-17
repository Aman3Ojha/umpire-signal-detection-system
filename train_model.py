import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Configuration
DATA_PATH = os.path.join('data')
ACTIONS = np.array(['Out', 'Six', 'Four', 'Wide', 'No_Ball', 'Idle'])
no_sequences = 30
sequence_length = 30
feature_length = 132 # 33 landmarks * 4 values

label_map = {label:num for num, label in enumerate(ACTIONS)}

print("Loading data from:", DATA_PATH)
sequences, labels = [], []

# Try to load the data. If it fails, inform the user they need to run data collection first.
try:
    for action in ACTIONS:
        # Sort sequences to ensure consistent loading
        seq_files = os.listdir(os.path.join(DATA_PATH, action))
        seq_files = [f for f in seq_files if f.endswith('.npy')]
        
        for seq_file in seq_files:
            window = np.load(os.path.join(DATA_PATH, action, seq_file))
            sequences.append(window)
            labels.append(label_map[action])
except Exception as e:
    print(f"Error loading data: {e}")
    print("Please make sure you have run 'data_collection.py' to generate the training data first.")
    exit()

if not sequences:
    print("No data found. Please run 'data_collection.py' to collect data.")
    exit()

X = np.array(sequences)
y = to_categorical(labels).astype(int)

print(f"Data Shape: X={X.shape}, y={y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print("Building the LSTM model...")
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(sequence_length, feature_length)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(ACTIONS.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.summary()

# Checkpoint to save the best model during training
checkpoint = ModelCheckpoint('umpire_model.keras', monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')

print("Starting model training...")
# Train the model
model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[checkpoint])

# Optional: Save the last epoch's weights as well
model.save('umpire_model_final.keras')
print("Model training complete! The best model is saved as 'umpire_model.keras'")
