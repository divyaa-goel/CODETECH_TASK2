NAME : DIVYA MANISH GOEL
DURATION : JULY 15TH - SEPTEMBER 15TH 2024
ID : CT8ML1842
COURSE: MACHINE LEARNIHG
OVERVIEW OF THIS PROJECT: 
Objective: Build a sentiment analysis model using TensorFlow to classify IMDb movie reviews as positive or negative.

1. Import Necessary Libraries
You start by importing essential libraries for the project:

tensorflow for building and training the neural network.
tensorflow_datasets for loading the IMDb dataset.
sklearn for potentially splitting datasets, though it's not used in this version.
keras components for creating the neural network layers.
2. Load IMDb Dataset
tfds.load('imdb_reviews', with_info=True, as_supervised=True) loads the IMDb movie reviews dataset.
The dataset is split into training and test datasets.
3. Data Preprocessing
Shuffle: Randomize the order of the training data to ensure the model doesn't learn any unintended order.
Batching: Group the data into batches for efficient training. The padded_batch function is used to ensure all sequences in a batch are the same length.
4. Build the Model
Embedding Layer: Converts integer word indices into dense vectors of fixed size (16 dimensions in this case).
GlobalAveragePooling1D: Reduces the sequence dimension by averaging over all the embeddings.
Dense Layers: Two fully connected layers:
A hidden layer with ReLU activation.
An output layer with sigmoid activation for binary classification.
5. Compile the Model
Optimizer: Adam optimizer for adjusting weights during training.
Loss Function: Binary cross-entropy to measure the error in binary classification.
Metrics: Accuracy to evaluate the performance of the model.
6. Train the Model
Training: Fit the model on the training data for 10 epochs, validating on the test data.
Validation: Evaluate the model's performance on a separate test set to check how well it generalizes.
7. Evaluate the Model
Evaluation: Measure the final performance of the model on the test dataset, printing the test accuracy.
