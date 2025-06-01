import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import register_keras_serializable
from collections import Counter
from typing import List

"""
# Data processing - Read CSV - Input format: word, tag
df = pd.read_csv("ner_dataset_formatted.csv", header=None)

X = []
y = []

current_words = []
current_tags = []

for i, row in df.iterrows():
    word = str(row[0]).strip() if pd.notna(row[0]) else ""
    tag = str(row[1]).strip() if pd.notna(row[1]) else ""

    if word == ".":
        if current_words:
            X.append(current_words)
            y.append(current_tags)
            current_words = []
            current_tags = []
    elif word and tag:  # Skip blanks
        current_words.append(word)
        current_tags.append(tag)

# Handle any trailing sentence
if current_words:
    X.append(current_words)
    y.append(current_tags)

# Save the processed data to a pickle file    
with open("ner_data.pkl", "wb") as f:
    pickle.dump((X, y), f)

"""


# Pad sequences to ensure uniform input size for the model
def pad_sequences( inputs: List[List[int]], tags: List[List[int]], pad_token_id: int = 0, pad_tag_id: int = 0):
    max_len = max(len(seq) for seq in inputs)
    
    padded_inputs = []
    padded_tags = []
    
    for input_ids, tag_ids in zip(inputs, tags):
        input_padding = [pad_token_id] * (max_len - len(input_ids))
        
        tag_padding = [pad_tag_id] * (max_len - len(tag_ids))
        
        padded_inputs.append(input_ids + input_padding)
        padded_tags.append(tag_ids + tag_padding)
    
    return padded_inputs, padded_tags

# Load the preprocessed data
with open("ner_data.pkl", "rb") as f:
    X, y = pickle.load(f)

# Flatten list of words
all_words = [word for sentence, tags in zip(X, y)
                for word, tag in zip(sentence, tags) if tag == "O"]

# Count word frequencies (optional but useful for vocab trimming)
word_freq = Counter(all_words)

# Build word2idx (starting index from 2 to reserve 0 for PAD and 1 for UNK)
word2idx = {word: idx + 2 for idx, word in enumerate(word_freq.keys())}
word2idx["<PAD>"] = 0
word2idx["<UNK>"] = 1

# Inverse mapping
idx2word = {idx: word for word, idx in word2idx.items()}

# Get all unique tags
all_tags = sorted(set(tag for tags in y for tag in tags))

tag2idx = {tag: idx + 1 for idx, tag in enumerate(all_tags)}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}

X_ids = []
y_ids = []

for sentence_tokens, sentence_labels in zip(X, y):
    sentence_word_ids = [word2idx.get(token, word2idx.get('[UNK]', 1)) for token in sentence_tokens]
    sentence_label_ids = [tag2idx[label] for label in sentence_labels]
    
    X_ids.append(sentence_word_ids)
    y_ids.append(sentence_label_ids)

padded_inputs, padded_tags = pad_sequences(X_ids, y_ids)

# # vocab size - 23796
# # tag size - 17

# Get positional encoding for Transformer
def get_positional_encoding(seq_len, embed_dim):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(embed_dim)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
    angle_rads = pos * angle_rates

    # apply sin to even indices in the array; cos to odd indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


@register_keras_serializable()
class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def build(self, input_shape):
        super().build(input_shape)


def build_transformer_ner(
    vocab_size,           # input vocab size
    max_len,              # max sequence length
    embed_dim=128,        # embedding dimension
    num_heads=4,          # multi-head attention heads
    ff_dim=512,           # feed-forward layer dimension
    num_layers=2,         # number of Transformer encoder blocks
    num_tags=10,          # number of NER tags (output classes)
    dropout_rate=0.1
):
    inputs = Input(shape=(max_len,))

    # Token embeddings
    embedding_layer = Embedding(vocab_size, embed_dim, mask_zero=True)(inputs)

    # Add positional encoding
    pos_encoding = get_positional_encoding(max_len, embed_dim)
    x = embedding_layer + pos_encoding

    # Apply multiple Transformer Encoder layers
    for _ in range(num_layers):
        x = TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)

    # Output layer for token classification
    # Use TimeDistributed Dense for per-token tag predictions
    outputs = Dense(num_tags, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
    
model = build_transformer_ner(
    vocab_size=23796,
    max_len=103,
    embed_dim=128,
    num_heads=4,
    ff_dim=512,
    num_layers=2,
    num_tags=18  # adjust to your NER tags count
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Devices: ", tf.config.list_physical_devices())
tf.debugging.set_log_device_placement(True)

model.fit(
    x=np.array(padded_inputs),           # shape: (num_samples, max_len)
    y=np.array(padded_tags),             # shape: (num_samples, max_len)
    batch_size=32,
    epochs=5,
    validation_split=0.1
)

model.save("ner_transformer.keras")

loaded_model = load_model("ner_transformer.keras")
loaded_model.summary()

max_len = 103
pad_token_id = word2idx["<PAD>"]

# Function to tokenize and pad sentences for inference
def tokenize_and_pad(sentences, max_len=max_len, pad_token_id=pad_token_id):
    all_ids = []
    for sentence in sentences:
        tokens = sentence.split()  
        ids = [word2idx.get(token, word2idx.get('<UNK>', 1)) for token in tokens]
        if len(ids) < max_len:
            ids += [pad_token_id] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        all_ids.append(ids)
    return np.array(all_ids)

# Example sentences for inference
example_sentences = [
    "Sentence - 1",
    "Sentence - 2",
    "Sentence - 3",
    "Sentence - 4",
    "Sentence - 5"
]

padded_example_Xids = tokenize_and_pad(example_sentences)
predictions = loaded_model.predict(padded_example_Xids)
predicted_tag_ids = np.argmax(predictions, axis=-1)

print("Predicted tag IDs:", predicted_tag_ids)
for i, sentence_ids in enumerate(predicted_tag_ids):
    print(f"\nSentence {i + 1} tags:")

    words = example_sentences[i].split()  # tokenize sentence into words
    for word, tag_id in zip(words, sentence_ids):
        if tag_id != 0:  # skip padded tokens
            print(f"{word} → {idx2tag[tag_id]}")
