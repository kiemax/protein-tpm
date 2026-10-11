import tensorflow as tf
from keras import Model, layers
from tensorflow import keras

####-------------------------------------------------------------------simple dummy model


def build_simple_multi_input_model(channel1_shape, channel2_shape):

    # Example if you have 3 input channels
    input1 = layers.Input(shape=channel1_shape)
    input2 = layers.Input(shape=channel2_shape)
    # input3 = layers.Input(shape=channel3_shape)

    x1_layer = layers.Dense(80, activation="relu", name="dense_input1")
    x1 = x1_layer(input1)

    x2_layer = layers.Dense(80, activation="relu", name="dense_input2")
    x2 = x2_layer(input2)

    # x3_layer = layers.Dense(80, activation='relu', name = "dense_input3")
    # x3 = x3_layer(input3)

    f1 = layers.Flatten()(x1)
    f2 = layers.Flatten()(x2)
    # f3 = layers.Flatten()(x3)

    concat = layers.Concatenate()([f1, f2])
    dense_layer = layers.Dense(64, activation="relu", name="dense_concat")
    dense = dense_layer(concat)

    output_layer = layers.Dense(1, activation="sigmoid", name="output")
    output = output_layer(dense)

    model = Model(inputs=[input1, input2], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


def build_simple_multi_input_model_with_norm(norm1, channel1_shape, channel2_shape):
    """
    norm1: a keras.layers.Normalization that’s already been adapted
    channel1_shape = (maxlen, n_feats1)
    channel2_shape = (n_feats2,)  # one‐hot length
    """
    # ─── Inputs ──────────────────────────────────────────────────────────────────
    input1 = layers.Input(shape=channel1_shape, name="input1")
    input2 = layers.Input(shape=channel2_shape, name="input2")  # one‐hot

    # ─── Normalize only input1 ──────────────────────────────────────────────────
    x1_normed = norm1(input1)  # now shape (batch, maxlen, n_feats1)

    # ─── Dense on normalized input1 ─────────────────────────────────────────────
    x1 = layers.Dense(80, activation="relu", name="dense_input1")(x1_normed)

    # ─── Dense on input2 (no normalization) ────────────────────────────────────
    x2 = layers.Dense(80, activation="relu", name="dense_input2")(input2)
    #   Since input2 is one‐hot, a Dense layer can learn weights directly.

    # ─── Flatten & Concatenate ─────────────────────────────────────────────────
    f1 = layers.Flatten()(x1)
    f2 = layers.Flatten()(x2)  # (batch,) vector flatten, no change really
    merged = layers.Concatenate()([f1, f2])

    # ─── Downstream dense & output ──────────────────────────────────────────────
    dense_concat = layers.Dense(64, activation="relu", name="dense_concat")(merged)
    output_layer = layers.Dense(1, activation="sigmoid", name="output")(dense_concat)

    model = Model(inputs=[input1, input2], outputs=output_layer)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def create_ffn(hidden_units, dropout_rate, final_dropout=True):
    ffn_layers = []
    for units in hidden_units[:-1]:
        ffn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    ffn_layers.append(layers.Dense(units=hidden_units[-1]))
    if final_dropout:
        ffn_layers.append(layers.Dropout(dropout_rate))
    ffn = keras.Sequential(ffn_layers)
    return ffn



class TokenAndPositionEmbedding(layers.Layer):
    """
    emebd_dim = projection_dim
    maxlen = num_patches...
    vocab_size = ? no tokens in image perceiver, already vectors it seems
    """

    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    # Override get_config to make the layer serializable
    def get_config(self):
        config = super().get_config()
        config.update(
            {"maxlen": self.maxlen, "vocab_size": self.vocab_size, "embed_dim": self.embed_dim}
        )
        return config

    # Optional: from_config method to allow reloading from saved config
    @classmethod
    def from_config(cls, config):
        return cls(**config)


def create_cross_attention_module(latent_dim, data_dim, projection_dim, ffn_units, dropout_rate):
    inputs = {
        # Recieve the latent array as an input of shape [1, latent_dim, projection_dim].
        "latent_array": layers.Input(shape=(latent_dim, projection_dim)),
        # Recieve the data_array (encoded image) as an input of shape [batch_size, data_dim, projection_dim].
        "data_array": layers.Input(shape=(data_dim, projection_dim)),
    }
    # Apply layer norm to the inputs
    latent_array = layers.LayerNormalization(epsilon=1e-6)(inputs["latent_array"])
    data_array = layers.LayerNormalization(epsilon=1e-6)(inputs["data_array"])

    # Create query tensor: [1, latent_dim, projection_dim].
    query = layers.Dense(units=projection_dim)(latent_array)
    # Create key tensor: [batch_size, data_dim, projection_dim].
    key = layers.Dense(units=projection_dim)(data_array)
    # Create value tensor: [batch_size, data_dim, projection_dim].
    value = layers.Dense(units=projection_dim)(data_array)
    # Generate cross-attention outputs: [batch_size, latent_dim, projection_dim].
    attention_output = layers.Attention(use_scale=True, dropout=0.1)(
        [query, key, value], return_attention_scores=False
    )
    # Skip connection 1.
    attention_output = layers.Add()([attention_output, latent_array])
    # Apply layer norm.
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output)
    # Apply Feedforward network.
    ffn = create_ffn(hidden_units=ffn_units, dropout_rate=dropout_rate)
    outputs = ffn(attention_output)
    # Skip connection 2.
    outputs = layers.Add()([outputs, attention_output])
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def create_transformer_module(
    latent_dim,
    projection_dim,
    num_heads,
    num_transformer_blocks,
    ffn_units,
    dropout_rate,
):
    # projection_dim is embed_dim in text Transformer
    # input_shape: [1, latent_dim, projection_dim]
    inputs = layers.Input(shape=(latent_dim, projection_dim))

    x0 = inputs
    # Create multiple layers of the Transformer block.
    for _ in range(num_transformer_blocks):
        # Apply layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(x0)
        # Create a multi-head self-attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1. the identity is preserved due to a backpropagation through the identity function
        x2 = layers.Add()([attention_output, x0])
        # Apply layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # Apply Feedforward network.
        ffn = create_ffn(hidden_units=ffn_units, dropout_rate=dropout_rate)
        x3 = ffn(x3)
        # Skip connection 2.
        x0 = layers.Add()([x3, x2])

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=x0)
    return model


@tf.keras.utils.register_keras_serializable()
class Perceiver(keras.Model):
    def __init__(
        self,
        data_dim,
        vocab_size,
        latent_dim,
        projection_dim,
        num_heads,
        num_transformer_blocks,
        ffn_units,
        dropout_rate,
        num_iterations,
        classifier_units,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks
        self.ffn_units = ffn_units
        self.dropout_rate = dropout_rate
        self.num_iterations = num_iterations
        self.classifier_units = classifier_units

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "data_dim": self.data_dim,
                "vocab_size": self.vocab_size,
                "latent_dim": self.latent_dim,
                "projection_dim": self.projection_dim,
                "num_heads": self.num_heads,
                "num_transformer_blocks": self.num_transformer_blocks,
                "ffn_units": self.ffn_units,
                "dropout_rate": self.dropout_rate,
                "num_iterations": self.num_iterations,
                "classifier_units": self.classifier_units,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        # Create latent array.
        self.latent_array = self.add_weight(
            shape=(self.latent_dim, self.projection_dim),
            initializer="random_normal",
            trainable=True,
        )
        # print('Build Embedding')
        # self.embedding_layer = TokenAndPositionEmbedding(self.data_dim, self.vocab_size, self.projection_dim)

        # self.concatenating_layer =  tf.keras.layers.concatenate()
        # Create cross-attenion module.
        self.cross_attention = create_cross_attention_module(
            self.latent_dim,
            self.data_dim,
            self.projection_dim,
            self.ffn_units,
            self.dropout_rate,
        )

        # Create Transformer module.
        self.transformer = create_transformer_module(
            self.latent_dim,
            self.projection_dim,
            self.num_heads,
            self.num_transformer_blocks,
            self.ffn_units,
            self.dropout_rate,
        )

        # Create global average pooling layer.
        self.global_average_pooling = layers.GlobalAveragePooling1D()
        # Create a classification head.
        self.classification_head = create_ffn(
            hidden_units=self.classifier_units, dropout_rate=self.dropout_rate,
            final_dropout=False,
        )
        super().build(input_shape)

    def call(self, inputs):
        cross_attention_inputs = {
            "latent_array": tf.expand_dims(self.latent_array, 0),
            "data_array": inputs,
        }
        # Apply the cross-attention and the Transformer modules iteratively.
        for _ in range(self.num_iterations):
            # Apply cross-attention from the latent array to the data array.
            latent_array = self.cross_attention(cross_attention_inputs)
            # Apply self-attention Transformer to the latent array.
            latent_array = self.transformer(latent_array)
            # Set the latent array of the next iteration.
            cross_attention_inputs["latent_array"] = latent_array

        # Apply global average pooling to generate a [batch_size, projection_dim] repesentation tensor.
        representation = self.global_average_pooling(latent_array)
        # Generate logits.
        logits = self.classification_head(representation)
        # perceiver = keras.Model(inputs=[input0, input1],outputs= logits)
        return logits


def create_model(
    normlayer,
    maxlen=25,  # Number of tokens per sample (e.g., atoms per molecule)
    num_atom_types=22,  # Vocabulary size for atom type tokens (e.g., 22 distinct atom types)
    embedding_dim=32,  # Size of the learned embedding vector for each token
    dist_feat_dim=3,  # Feature dimensionality of the distance vectors (per token)
    latent_dim=32,  # Number of latent tokens (internal "hidden" representation)
    num_heads=4,  # Number of attention heads in the Transformer block
    num_transformer_blocks=4,  # Number of stacked self-attention blocks
    num_iterations=2,  # Number of cross-attention + transformer iterations
    dropout_rate=0.2,  # Dropout regularization rate
    output_dim=1,  # Output size (1 for binary OvA classification)
):
    # ---------- Inputs ----------
    input_dist = layers.Input(shape=(maxlen, dist_feat_dim), name="distance_vectors")
    input_tokens = layers.Input(shape=(maxlen,), dtype="int32", name="atom_types")

    # ---------- Normalization Layers ----------
    # norm_layer_dist = layers.Normalization(name="norm_dist")
    norm_dist = normlayer(input_dist)

    # norm_chart = layers.Normalization(name="norm_chart")(input_chart)
    # You will call .adapt() on these layers outside the model (see below)

    # ---------- Embedding ----------
    embedding_layer = TokenAndPositionEmbedding(
        maxlen=maxlen, vocab_size=num_atom_types, embed_dim=embedding_dim
    )
    embedded_tokens = embedding_layer(input_tokens)  # Shape: (batch, maxlen, embedding_dim)

    # ---------- Merge Channels 1 & 2 ----------
    merged_main = layers.Concatenate(axis=-1)(
        [norm_dist, embedded_tokens]
    )  # Shape: (batch, maxlen, dist+embed_dim)

    projection_dim = dist_feat_dim + embedding_dim  # Final per-token vector length
    token_count = maxlen

    # ---------- FFN and Classifier Units ----------
    # ffn_units = [projection_dim]               # Feedforward layer inside attention (can be deeper)
    ffn_units = [128, projection_dim]  # Feedforward layer inside attention (can be deeper)
    classifier_units = [128, 64, output_dim]  # Final classification head

    # ---------- Perceiver Backbone ----------
    perceiver_block = Perceiver(
        data_dim=token_count,
        vocab_size=num_atom_types,  ## unused
        latent_dim=latent_dim,
        projection_dim=projection_dim,
        num_heads=num_heads,
        num_transformer_blocks=num_transformer_blocks,
        ffn_units=ffn_units,
        dropout_rate=dropout_rate,
        num_iterations=num_iterations,
        classifier_units=classifier_units,
    )

    # perceiver_output = perceiver_block(merged_tokens)
    perceiver_output = perceiver_block(merged_main)

    # ---------- Final Output ----------
    output_activation = layers.Activation("sigmoid", name="output_activation")
    output = output_activation(perceiver_output)

    # ---------- Final Output ----------
    # logits = perceiver_block(merged_tokens)
    # output = layers.Activation("sigmoid", name="output")(logits)

    return Model(inputs=(input_dist, input_tokens), outputs=output)
