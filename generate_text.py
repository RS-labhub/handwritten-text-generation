import tensorflow as tf
import numpy as np
import pandas as pd

# Load the dataset
csv_file_path = 'dataset/dataset.csv'
data = pd.read_csv(csv_file_path)

# Combine all text into a single string
text = ' '.join(data['text'].astype(str).tolist())

# Create vocabulary and mapping from characters to indices
vocab = sorted(set(text))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# Function to rebuild the model with the same architecture
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

# Load the trained model
checkpoint_dir = './checkpoints'
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(len(vocab), 256, 1024, batch_size=1)
model.load_weights(latest_checkpoint)
model.build(tf.TensorShape([1, None]))

# Function to generate text
def generate_text(model, start_string):
    num_generate = 1000  # Number of characters to generate
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

# Generate text using the trained model
start_string = "Once upon a time"
print(generate_text(model, start_string))
