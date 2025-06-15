"""Modul Transform untuk melakukan preprocessing data."""

import tensorflow as tf


LABEL_KEY = "status"
FEATURE_KEY = "statement"


def transformed_name(key: str) -> str:
    """Mengembalikan nama fitur yang telah ditransformasi."""
    return f"{key}_xf"


def preprocessing_fn(inputs: dict) -> dict:
    """Melakukan preprocessing pada fitur mentah."""
    outputs = {}

    label_table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=[
                "Anxiety",
                "Bipolar",
                "Depression",
                "Normal",
                "Personality disorder",
                "Stress",
                "Suicidal",
            ],
            values=[0, 1, 2, 3, 4, 5, 6],
            key_dtype=tf.string,
            value_dtype=tf.int64,
        ),
        default_value=-1,
    )

    outputs[transformed_name(LABEL_KEY)] = label_table.lookup(inputs[LABEL_KEY])
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])

    return outputs
