"""
https://github.com/owahltinez/triplet-loss-animal-reid/blob/main/experiment.py

Trains a model using the triplet learning architecture.
"""
import json
import time
import pathlib
import shutil
import tempfile
import argparse

import keras_cv
import numpy as np
import torch
from torchvision import transforms

import dataset
from utils import mean_average_precision

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Parameters")

    # I/O parameters
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to image dataset, either locally available or in GCS.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Zip file where experiment results will be stored.",
    )

    # Hyper-parameters for transfer learning fine-tuning
    parser.add_argument("--batch_size", type=int, default=32, help="Number of samples per batch.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Rate of learning during training.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout regularization factor applied during training.")
    parser.add_argument("--augmentation_count", type=int, default=4, help="Number of augmentations to apply per image.")
    parser.add_argument("--augmentation_factor", type=float, default=0.1, help="Factor of image augmentation.")
    parser.add_argument("--loss_margin", type=float, default=0.5, help="Margin used for semi-hard triplet loss.")
    parser.add_argument("--embedding_size", type=int, default=128, help="Output embedding dimensions.")
    parser.add_argument("--retrain_layer_count", type=int, default=128, help="Number of layers to retrain from base model.")

    # Experiment-related parameters
    parser.add_argument("--train_epochs", type=int, default=50, help="Total training epochs.")
    parser.add_argument("--save_model", action="store_true", help="Whether to save the trained model.")
    parser.add_argument("--vote_count", type=int, default=5, help="Number of votes per embedding in closed eval mode.")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level used for the model fit.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for various random number generators.")

    return parser.parse_args()

INPUT_SHAPE = (224, 224, 3)


def _preprocessing_layer(x: tf.keras.layers.Input):
  x = tf.cast(x, tf.float32)
  x = tf.keras.applications.densenet.preprocess_input(x)
  return x


def _augmentation_layer(
    x: tf.keras.layers.Input,
    augmentation_count: int,
    augmentation_factor: float,
):
  n = augmentation_count
  k = augmentation_factor

  # Early exit: no augmentations necessary.
  if n == 0 or k < 1e-6:
    return x

  t_opts = dict(seed=args.seed)
  transformations = [
      tf.keras.layers.Dropout(k, **t_opts),
      tf.keras.layers.GaussianNoise(k, **t_opts),
      tf.keras.layers.RandomFlip("horizontal", **t_opts),
      tf.keras.layers.RandomTranslation(k, k, fill_mode="constant", **t_opts),
      tf.keras.layers.RandomRotation(k, fill_mode="constant", **t_opts),
      tf.keras.layers.RandomZoom(k, k, fill_mode="constant", **t_opts),
      keras_cv.layers.RandomHue(k, [0, 255], **t_opts),
      keras_cv.layers.RandomSaturation(k, **t_opts),
  ]

  for t in np.random.choice(transformations, size=n, replace=False):
    x = t(x)

  return x


def _inference_layer(
    x: tf.keras.layers.Input,
    dropout: float,
    embedding_size: int,
):
  # Used to initialize the weights of untrained layers.
  w = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")

  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  if dropout > 0:
    x = tf.keras.layers.Dropout(dropout)(x)
  x = tf.keras.layers.Dense(embedding_size, activation=None, bias_initializer=w)(x)
  x = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="embedding")(x)

  return x


def _transfer_layer(
    x: tf.keras.layers.Input,
    retrain_layer_count: int,
):
  model = tf.keras.applications.DenseNet121(
      include_top=False,
      weights="imagenet",
      input_shape=INPUT_SHAPE,
  )

  # Enable training only for the selected layers of the base model.
  for layer in model.layers[:-retrain_layer_count]:
    layer.trainable = False

  # Run the input through the model.
  x = model(x)

  return x


def _build_model(
    dropout: float,
    augmentation_count: int,
    augmentation_factor: float,
    embedding_size: int,
    retrain_layer_count: int,
    loss_margin: float,
    learning_rate: float,
) -> tf.keras.Model:
  # Initialize the input parameters.
  x = inputs = tf.keras.layers.Input(shape=(None, None, 3), dtype=tf.uint8)

  # Run the inputs through the model's layers.
  x = _preprocessing_layer(x)
  x = _augmentation_layer(x, augmentation_count, augmentation_factor)
  x = _transfer_layer(x, retrain_layer_count)
  x = _inference_layer(x, dropout, embedding_size)

  # Encapsulate the I/O into a model type.
  model = tf.keras.Model(inputs=[inputs], outputs=x)

  # Compile the model and return it.
  model.compile(
      loss=tfa.losses.TripletSemiHardLoss(soft=False, margin=loss_margin),
      optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
  )

  return model


def main(args):
  # Validate required parameters.
  if not str(args.output).endswith(".zip"):
    raise ValueError(f'Parameter "output" must be a path to a zip file: {args.output}.')

  # Create a folder to save progress and results.
  output_root = pathlib.Path(tempfile.mkdtemp())
  experiment_path = output_root / "experiment"
  experiment_path.mkdir(parents=True, exist_ok=True)
  print(f"Using {output_root} to store temporary experiment results.")

  # Download the dataset if it doesn't exist locally.
  dataset_name = args.dataset.split("/")[-1]
  if pathlib.Path(args.dataset).exists():
    dataset_path = pathlib.Path(args.dataset)
    print("Dataset %s found in local disk.", dataset_name)
  elif args.dataset.startswith("gs://"):
    dataset_path = output_root / dataset_name
    print("Downloading dataset %s into %s.", dataset_name, str(dataset_path))
    bucket_name = args.dataset[len("gs://") :].split("/", 1)[0]
    dataset.download_from_gcloud(
        bucket_name=bucket_name,
        output_dir=dataset_path.parent,
        prefix=mean_average_precision.dataset[len("gs://") + len(bucket_name) + 1 :] + "/",
    )
  else:
    raise ValueError(f"Unknown dataset location: {FLAGS.dataset}")

  # Retrieve parameters from flags.
  parameters = dict(
      dropout=args.dropout,
      augmentation_count=args.augmentation_count,
      augmentation_factor=args.augmentation_factor,
      learning_rate=args.learning_rate,
      embedding_size=args.embedding_size,
      retrain_layer_count=args.retrain_layer_count,
      loss_margin=args.loss_margin,
  )

  # Start experiment.
  time_start = time.perf_counter()
  print(f"Starting experiment for {dataset_name}.")
  results = dict(dataset=dataset_name)

  # Build and compile model using provided parameters.
  model = _build_model(**parameters)

  # Add non-model parameters to the parameter dictionary.
  parameters["dataset"] = dataset_name
  parameters["batch_size"] = args.batch_size
  parameters["train_epochs"] = args.train_epochs

  # Load dataset from filesystem.
  if (dataset_path / "train").exists() and (dataset_path / "test").exists():
    # Datasets might already be split between train and test.
    ds_opts = dict(
        image_size=list(INPUT_SHAPE)[:-1],
        batch_size=args.batch_size,
        seed=args.seed,
    )
    train_data = dataset.triplet_safe_image_dataset_from_directory(
        dataset_path / "train",
        **ds_opts,
    )
    test_data = tf.keras.utils.image_dataset_from_directory(
        dataset_path / "test",
        **ds_opts,
    )
    validation_data = (
        tf.keras.utils.image_dataset_from_directory(
            dataset_path / "validation",
            **ds_opts,
        )
        # If there's no specific validation subset, re-use test data.
        if (dataset_path / "validation").exists()
        else test_data
    )

  else:
    # Otherwise we will need to create our own test subset.
    train_data, validation_data, test_data = dataset.load_dataset(
        dataset_path,
        [0.7, 0.1, 0.2],
        image_size=list(INPUT_SHAPE)[:-1],
        batch_size=args.batch_size,
        seed=args.seed,
    )

  # Determine if it's closed or open eval based on whether all labels from the
  # training set are also present in the test set.
  if all(label in test_data.class_names for label in train_data.class_names):
    model_eval = "closed"
  else:
    model_eval = "open"
  parameters["model_eval"] = model_eval
  print("Using model evaluation: %s set.", model_eval)

  # Write the subset labels to disk.
  datasets = train_data, validation_data, test_data
  for name, subset in zip(["train", "validation", "test"], datasets):
    if subset is not None:
      subset_labels = subset.class_names
      print("Using %d classes for subset %s.", len(subset_labels), name)
      with open(experiment_path / f"{name}_subset.txt", "w") as f:
        f.write("\n".join({label for label in subset_labels}))

  # Write out the model's summary to disk.
  print("Writing model summary.")
  with open(experiment_path / "model_summary.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))
    for layer in model.layers:
      if hasattr(layer, "summary"):
        layer.summary(print_fn=lambda x: f.write(x + "\n"))

  # Fit the model for the entire number of epochs unless early stopping.
  callbacks = [
      tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
      tf.keras.callbacks.TensorBoard(log_dir=experiment_path / f"log"),
  ]

  # Compute the mAP each epoch using open set eval mode only (faster).
  cb_subset = validation_data if validation_data else test_data
  mAP_callback = mean_average_precision.MeanAveragePrecisionCallback
  callbacks.append(mAP_callback(top_k=[1, 5], dataset=cb_subset))

  # Fit the model.
  history = model.fit(
      x=train_data.cache().prefetch(tf.data.AUTOTUNE),
      validation_data=validation_data.cache().prefetch(tf.data.AUTOTUNE),
      epochs=args.train_epochs,
      verbose=args.verbose,
      callbacks=callbacks,
  )

  # Write out the used parameters.
  with open(experiment_path / "parameters.json", "w") as f:
    json.dump(parameters, f)

  # Save the total number of epochs run.
  results["epochs"] = len(history.history["loss"])

  # Record into results the last metric values from the history.
  for key in history.history.keys():
    results[key] = history.history[key][-1]

  # Compute elapsed time for training loop.
  results["time_train"] = time.perf_counter() - time_start

  # Compute mean average precision metrics and save them to disk.
  map_top_k = (1, 5)
  print("Computing mAP@%r using %s set eval.", map_top_k, model_eval)
  if model_eval == "open":
    results.update(
        mean_average_precision.evaluate_model_open_set(
            model,
            test_data,
            top_k=map_top_k,
        )
    )
  elif model_eval == "closed":
    results.update(
        mean_average_precision.evaluate_model_closed_set(
            model,
            train_data,
            test_data,
            top_k=map_top_k,
            vote_count=args.vote_count,
        )
    )

  # Compute elapsed time for entire trial, including evaluation.
  results["time_trial"] = time.perf_counter() - time_start
  results["time_eval"] = results["time_trial"] - results["time_train"]

  # Log results and save them to disk.
  print("Experiment %s results: %r", dataset_name, results)
  with open(experiment_path / "results.json", "w") as f:
    json.dump(results, f)

  # Save the model into a file for later analysis.
  if args.save_model:
    print("Saving model into output folder.")
    model.save(experiment_path / f"model")

  # Zip results and copy them to final output destination.
  print("Writing output to %s.", args.output)
  output_tmp_zip = f"{experiment_path}.zip"
  shutil.make_archive(experiment_path, "zip", experiment_path)
  with (
      open(output_tmp_zip, "rb") as f_in,
      tf.io.gfile.GFile(args.output, "wb") as f_out,
  ):
    shutil.copyfileobj(f_in, f_out)

  # Remove temporary files.
  print("Removing temporary files at %s.", output_root)
  shutil.rmtree(output_root)


if __name__ == "__main__":
  args = parse_args()
  main(args)