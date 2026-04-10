import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def collect_train_samples(train_dir: Path):
    image_paths = sorted(train_dir.glob("*.jpg"))
    samples = []
    for path in image_paths:
        name = path.name.lower()
        if name.startswith("cat."):
            label = 0
        elif name.startswith("dog."):
            label = 1
        else:
            continue
        samples.append((str(path), label))
    return samples


def build_model(input_shape=(224, 224, 3), train_base=False):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = train_base

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_history_plots(history, output_dir: Path):
    history_path = output_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="Train Accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def write_summary(model, history, output_dir: Path):
    val_acc = max(history.history.get("val_accuracy", [0.0]))
    train_acc = max(history.history.get("accuracy", [0.0]))
    summary = {
        "best_train_accuracy": float(train_acc),
        "best_val_accuracy": float(val_acc),
        "epochs_trained": int(len(history.history.get("loss", []))),
        "model_path": str((output_dir / "best_model.keras").resolve()),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (output_dir / "model_summary.txt").open("w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    return val_acc


def predict_test_set(model, test_dir: Path, output_dir: Path, image_size=(224, 224), batch_size=64):
    test_images = sorted(test_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    submission_path = output_dir / "submission.csv"

    test_df = pd.DataFrame({"filename": [str(p) for p in test_images]})
    test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_flow = test_gen.flow_from_dataframe(
        test_df,
        x_col="filename",
        y_col=None,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,
    )

    preds = model.predict(test_flow, verbose=1).reshape(-1)

    with submission_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "label"])
        for img_path, pred in zip(test_images, preds):
            writer.writerow([img_path.stem, float(pred)])

    return submission_path


def main():
    parser = argparse.ArgumentParser(description="Dogs vs Cats CNN trainer")
    parser.add_argument("--train_dir", type=str, default="train", help="Path to train folder")
    parser.add_argument("--test_dir", type=str, default="test1", help="Path to test folder")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Folder for all outputs")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=12, help="Maximum epochs")
    parser.add_argument("--val_size", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fine_tune", action="store_true", help="Fine-tune the base model")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=10000,
        help="Maximum number of labeled training images to use (0 = all)",
    )
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="Retrain even if outputs/best_model.keras already exists",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "best_model.keras"
    run_summary_path = output_dir / "run_summary.json"

    if model_path.exists() and not args.force_retrain:
        print(f"Found existing trained model at: {model_path}")
        print("Skipping training. Use --force_retrain to train again.")
        model = tf.keras.models.load_model(model_path)

        input_height = int(model.input_shape[1])
        input_width = int(model.input_shape[2])
        prediction_size = (input_height, input_width)

        best_val_acc = 0.0
        if run_summary_path.exists():
            with run_summary_path.open("r", encoding="utf-8") as f:
                previous_summary = json.load(f)
                best_val_acc = float(previous_summary.get("best_val_accuracy", 0.0))
    else:
        samples = collect_train_samples(train_dir)
        if not samples:
            raise FileNotFoundError(f"No valid images found in {train_dir}.")

        if args.max_train_samples and args.max_train_samples > 0 and len(samples) > args.max_train_samples:
            # Keep a balanced random subset for faster CPU training.
            cats = [s for s in samples if s[1] == 0]
            dogs = [s for s in samples if s[1] == 1]
            half = args.max_train_samples // 2
            random.shuffle(cats)
            random.shuffle(dogs)
            samples = cats[:half] + dogs[:half]
            random.shuffle(samples)

        df = np.array(samples, dtype=object)
        x = df[:, 0]
        y = df[:, 1].astype(np.int32)

        x_train, x_val, y_train, y_val = train_test_split(
            x,
            y,
            test_size=args.val_size,
            random_state=args.seed,
            stratify=y,
        )

        train_df = pd.DataFrame({"filename": x_train, "class": y_train.astype(str)})
        val_df = pd.DataFrame({"filename": x_val, "class": y_val.astype(str)})

        train_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
        )
        val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

        train_flow = train_gen.flow_from_dataframe(
            train_df,
            x_col="filename",
            y_col="class",
            target_size=(args.img_size, args.img_size),
            batch_size=args.batch_size,
            class_mode="binary",
            shuffle=True,
            seed=args.seed,
        )

        val_flow = val_gen.flow_from_dataframe(
            val_df,
            x_col="filename",
            y_col="class",
            target_size=(args.img_size, args.img_size),
            batch_size=args.batch_size,
            class_mode="binary",
            shuffle=False,
        )

        model = build_model(input_shape=(args.img_size, args.img_size, 3), train_base=args.fine_tune)

        callbacks = [
            ModelCheckpoint(
                filepath=str(output_dir / "best_model.keras"),
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1,
            ),
            EarlyStopping(
                monitor="val_accuracy",
                patience=4,
                mode="max",
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.3,
                patience=2,
                min_lr=1e-7,
                verbose=1,
            ),
            CSVLogger(str(output_dir / "training_log.csv")),
        ]

        history = model.fit(
            train_flow,
            validation_data=val_flow,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1,
        )

        save_history_plots(history, output_dir)
        best_val_acc = write_summary(model, history, output_dir)
        prediction_size = (args.img_size, args.img_size)

    submission_path = predict_test_set(
        model,
        test_dir,
        output_dir,
        image_size=prediction_size,
        batch_size=max(64, args.batch_size),
    )

    print(f"\nBest validation accuracy: {best_val_acc * 100:.2f}%")
    print(f"Submission file saved to: {submission_path}")
    print(f"All outputs saved in: {output_dir.resolve()}")
    if best_val_acc >= 0.8:
        print("Target achieved: Validation accuracy is >= 80%.")
    else:
        print("Validation accuracy is below 80%. Try more epochs or --fine_tune.")


if __name__ == "__main__":
    main()
