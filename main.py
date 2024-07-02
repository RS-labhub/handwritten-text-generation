import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from PIL import Image, ImageDraw, ImageFont

# Load and preprocess the dataset
csv_file_path = 'dataset/dataset.csv'

# Load the dataset and print the first few rows to inspect
data = pd.read_csv(csv_file_path)
print(data.head())

# Check if 'IDENTITY' column exists and handle KeyError
if 'IDENTITY' not in data.columns:
    raise KeyError(f"'IDENTITY' column not found in the dataset. Available columns: {data.columns.tolist()}")

# Combine all text into a single string
text = ' '.join(data['IDENTITY'].astype(str).tolist())

# Create vocabulary and mapping from characters to indices
vocab = sorted(set(text))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# Convert text to integers
text_as_int = np.array([char2idx[c] for c in text])

# Create training examples and targets
seq_length = 100
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
dataset = sequences.map(lambda chunk: (chunk[:-1], chunk[1:]))

# Training parameters
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024
EPOCHS = 10

# Build and compile the model
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_shape=[None]),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform', batch_input_shape=[batch_size, None]),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)
model.compile(optimizer='adam', loss=lambda labels, logits: tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))

# Create a directory for saving checkpoints
checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_callback = ModelCheckpoint(filepath=os.path.join(checkpoint_dir, 'checkpoint_{epoch:02d}.h5'), save_weights_only=True, save_best_only=True, monitor='loss', verbose=1)

# Train the model
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

# Load the latest checkpoint
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(latest_checkpoint)
model.build(tf.TensorShape([1, None]))

# Function to generate text
def generate_text(model, start_string, num_generate=1000, temperature=1.0):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    return start_string + ''.join(text_generated)

# Generate text using the trained model
start_string = "Once upon a time"
generated_text = generate_text(model, start_string)
print(generated_text)

# Function to convert text to handwritten-like image
def text_to_image(text, font_path='QEBEV.ttf', font_size=30):
    font = ImageFont.truetype(font_path, font_size)
    lines = text.split('\n')
    max_width = max([font.getsize(line)[0] for line in lines])
    total_height = sum([font.getsize(line)[1] for line in lines])
    image = Image.new('RGB', (max_width, total_height), 'white')
    draw = ImageDraw.Draw(image)
    y_text = 0
    for line in lines:
        draw.text((0, y_text), line, font=font, fill='black')
        y_text += font.getsize(line)[1]
    image.show()

# Convert generated text to image
text_to_image(generated_text, font_path='QEBEV.ttf')
