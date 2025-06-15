"""Modul Tuner untuk pencarian hyperparameter."""

from typing import NamedTuple, Dict, Text, Any

import tensorflow as tf
import keras_tuner as kt
import tensorflow_transform as tft

from keras_tuner.engine import base_tuner
from tfx.components.trainer.fn_args_utils import FnArgs
from trainer import transformed_name, gzip_reader_fn


LABEL_KEY = "status"
FEATURE_KEY = "statement"
VOCAB_SIZE = 5000
MAX_TOKEN = 5000
SEQUENCE_LENGTH = 100
OUTPUT_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 16
NUM_EPOCHS = 5
BATCH_SIZE = 64
NUM_CLASSES = 7

TunerFnResult = NamedTuple(
    "TunerFnResult",
    [("tuner", base_tuner.BaseTuner),
     ("fit_kwargs", Dict[Text, Any])]
)


def input_fn(
    file_pattern: str,
    tf_transform_output: tft.TFTransformOutput,
    num_epochs: int,
    batch_size: int = BATCH_SIZE
) -> tf.data.Dataset:
    """Membuat dataset untuk tuning hyperparameter."""
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )
    return dataset


def model_builder(
    hp: kt.HyperParameters,
    vectorizer_layer: tf.keras.layers.Layer
) -> tf.keras.Model:
    """Membangun model Keras untuk KerasTuner."""
    inputs = tf.keras.Input(
        shape=(1,),
        name=transformed_name(FEATURE_KEY),
        dtype=tf.string
    )

    x = vectorizer_layer(inputs)
    x = tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(
        hp.Choice('unit_1', [64, 128, 256]),
        activation="relu"
    )(x)
    x = tf.keras.layers.Dropout(
        hp.Float('dropout_1', 0.1, 0.5, step=0.1)
    )(x)

    x = tf.keras.layers.Dense(
        hp.Choice('unit_2', [32, 64, 128]),
        activation="relu"
    )(x)
    x = tf.keras.layers.Dropout(
        hp.Float('dropout_2', 0.1, 0.5, step=0.1)
    )(x)

    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', [0.001, 0.0005, 0.0001])
        ),
        metrics=["accuracy"],
    )

    model.summary()
    return model


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Fungsi untuk menginisialisasi dan mengembalikan objek tuner serta parameter fit-nya."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = input_fn(
        fn_args.train_files[0],
        tf_transform_output,
        num_epochs=NUM_EPOCHS
    )

    eval_set = input_fn(
        fn_args.eval_files[0],
        tf_transform_output,
        num_epochs=NUM_EPOCHS
    )

    vectorizer_layer = tf.keras.layers.TextVectorization(
        max_tokens=MAX_TOKEN,
        output_mode="int",
        output_sequence_length=OUTPUT_SEQUENCE_LENGTH,
    )

    vectorizer_layer.adapt(
        train_set.map(lambda features, label: features[transformed_name(FEATURE_KEY)])
    )

    tuner = kt.RandomSearch(
        hypermodel=lambda hp: model_builder(hp, vectorizer_layer),
        objective="val_accuracy",
        max_trials=20,
        directory=fn_args.working_dir,
        project_name="mental_health_classifier",
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_set,
            "validation_data": eval_set,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
            "epochs": NUM_EPOCHS,
        }
    )
