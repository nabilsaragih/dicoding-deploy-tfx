import os
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
import json

LABEL_KEY = "status" 
FEATURE_KEY = "statement"  
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 64 
NUM_EPOCHS = 15 
BATCH_SIZE = 64
NUM_CLASSES = 7 

def transformed_name(key):
    return f"{key}_xf"

def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")

def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=BATCH_SIZE) -> tf.data.Dataset:
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )
    
    def _convert_labels(features, labels):
        return features, tf.cast(labels, tf.int32)
    
    return dataset.map(_convert_labels)

vectorize_layer = tf.keras.layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
)

def model_builder(hp):
    if isinstance(hp, dict) and 'values' in hp:
        hp = hp['values']

    inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
    reshaped_narrative = tf.reshape(inputs, [-1])
    
    x = vectorize_layer(reshaped_narrative)
    x = tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, name="embedding")(x)

    if hp.get('use_lstm', False):
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(hp.get('unit_1', 128), activation="relu")(x)
    x = tf.keras.layers.Dropout(hp.get('dropout_rate', 0.3))(x)
    x = tf.keras.layers.Dense(hp.get('unit_2', 64), activation="relu")(x)

    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x) 

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        loss="sparse_categorical_crossentropy", 
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.get('learning_rate', 0.001)
        ),
        metrics=["accuracy"], 
    )

    model.summary()
    return model

def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn

def run_fn(fn_args: FnArgs) -> None:
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq="batch"),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", verbose=1, patience=5
        ),
        tf.keras.callbacks.ModelCheckpoint(
            fn_args.serving_model_dir,
            monitor="val_accuracy",
            mode="max",
            verbose=1,
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, verbose=1)
    ]

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    train_set = input_fn(fn_args.train_files, tf_transform_output, NUM_EPOCHS)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, NUM_EPOCHS)

    vectorize_layer.adapt(
        train_set.map(lambda features, label: features[transformed_name(FEATURE_KEY)])
    )

    best_hyperparameters = fn_args.hyperparameters
    if isinstance(best_hyperparameters, str):
        best_hyperparameters = json.loads(best_hyperparameters)
    
    default_hparams = {
        'unit_1': 128,
        'unit_2': 64,
        'learning_rate': 0.001,
        'dropout_rate': 0.3,
        'use_lstm': False
    }
    best_hyperparameters = {**default_hparams, **best_hyperparameters}

    model = model_builder(best_hyperparameters)

    class_counts = tf.concat([y for x, y in train_set], axis=0)
    class_counts = tf.math.bincount(class_counts)
    class_weights = {i: 1.0/count for i, count in enumerate(class_counts.numpy())}
    class_weights = {k: v/min(class_weights.values()) for k, v in class_weights.items()}

    model.fit(
        x=train_set,
        validation_data=val_set,
        callbacks=callbacks,
        class_weight=class_weights, 
        steps_per_epoch=fn_args.train_steps or 1000,
        validation_steps=fn_args.eval_steps or 1000,
        epochs=NUM_EPOCHS,
    )

    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        )
    }

    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)