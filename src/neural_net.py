import os
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import Adam
from ml_utils import df_preprocessing
import numpy as np
from sklearn.preprocessing import StandardScaler

# Define the path to the data folder
data_folder = './data/RandomVsRandom'

# Initialize an empty dataframe to hold all the data
df = pd.DataFrame()

# Read and preprocess each CSV file in the folder
for filename in os.listdir(data_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_folder, filename)
        df_match = pd.read_csv(file_path)
        df_match = df_preprocessing(df_match)
        df = pd.concat([df, df_match], ignore_index=True)

# Separate features and labels
data = df.drop(columns=['number_of_turn', 'result']).values
# Normalize and center the data
scaler = StandardScaler()
data = scaler.fit_transform(data)
labels = df['result'].values

# Ensure labels are binary
labels = np.where(labels > 0.5, 1, 0)

# Verify data integrity
print(np.isnan(data).sum(), np.isinf(data).sum())

# Adjust optimizer
optimizer = Adam(learning_rate=0.0001)

# Initialize weights
model = Sequential()
model.add(Dense(64, input_dim=data.shape[1], activation='relu', kernel_initializer=HeNormal()))
for _ in range(6):
    model.add(Dense(64, activation='relu', kernel_initializer=HeNormal()))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Handle imbalance
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(class_weights))

# Split data manually
from sklearn.model_selection import train_test_split
data_train, data_val, labels_train, labels_val = train_test_split(data, labels, test_size=0.2, stratify=labels)

# Train model
model.fit(data_train, labels_train, epochs=50, batch_size=10, validation_data=(data_val, labels_val), class_weight=class_weights)

# Save the model
model.save('./model/nn_model0.keras')