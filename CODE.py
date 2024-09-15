# 1. Import necessary libraries
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Sequential

# 2. Load IMDb dataset
imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

# Split into training and test sets
train_data, test_data = imdb['train'], imdb['test']

# 3. Data Preprocessing (Prepare the data)
# Convert datasets to a format that can be used in training
BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_data = train_data.shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([None], []))
test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([None], []))

# 4. Build the model
embedding_dim = 16

model = Sequential([
    Embedding(input_dim=10000, output_dim=embedding_dim),  # Embedding layer
    GlobalAveragePooling1D(),  # Reduce sequence dimension
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# 5. Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 6. Train the model
history = model.fit(train_data, epochs=10, validation_data=test_data)

# 7. Evaluate the model
test_loss, test_acc = model.evaluate(test_data)
print(f'\nTest Accuracy: {test_acc}')
how to run this code on github
