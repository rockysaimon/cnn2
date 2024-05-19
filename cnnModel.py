import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
import numpy as np
import time
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.metrics import confusion_matrix

class MalwareDetection:
    def __init__(self, img_size, num_classes, batch_size, epochs, train_folder, test_folder):
        self.img_size = img_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.model = self.build_model()
        self.train_datagen = ImageDataGenerator(rescale=1. / 255)
        self.test_datagen = ImageDataGenerator(rescale=1. / 255)

    def build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(50, (2,2), input_shape=(self.img_size, self.img_size, 3)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(70, (3,3)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(70, (3,3)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(self.num_classes, activation="softmax")
        ])
        model.compile(
            loss="categorical_crossentropy", 
            optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, decay=0.001), 
            metrics=["accuracy"]
        )
        return model

    def create_class_weight(self, generator, mu=0.8):
        labels_dict = generator.classes
        total = len(generator.classes)
        unique_labels = np.unique(generator.classes)
        class_weight = dict()

        for label in unique_labels:
            class_count = np.sum(generator.classes == label)
            score = math.log(mu * total / float(class_count))
            class_weight[label] = score if score > 1.0 else 1.0

        return class_weight

    def train_model(self):
        training_set = self.train_datagen.flow_from_directory(
            self.train_folder,
            shuffle=True,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode="categorical"
        )

        test_set = self.test_datagen.flow_from_directory(
            self.test_folder,
            shuffle=False,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode="categorical"
        )

        class_weights = self.create_class_weight(training_set)

        model_file = "train_model/modelo3.keras"
        best_model = ModelCheckpoint(model_file, monitor="val_accuracy", save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        history = self.model.fit(
            training_set,
            validation_data=test_set,
            epochs=self.epochs,
            callbacks=[best_model, early_stopping],
            class_weight=class_weights
        )

        self.plot_training(history)
        self.evaluate_model(test_set, model_file)

    def plot_training(self, history):
        acc = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]

        actual_epochs = range(len(acc))

        plt.plot(actual_epochs, acc, "r", label="train_acc")
        plt.plot(actual_epochs, val_acc, "b", label="val_acc")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.title("train and val accuracy")
        plt.legend()
        plt.show()

    def evaluate_model(self, test_set, model_file):
        model = load_model(model_file)
        classes = list(test_set.class_indices.keys())

        self.calculate_and_plot_f1_score(test_set, classes, model)
        self.plot_confusion_matrix(test_set, model, classes)

    def calculate_and_plot_f1_score(self, test_data, classes, model):
        test_labels = test_data.labels
        predictions = model.predict(test_data)
        report_str = classification_report(test_labels, np.argmax(predictions, axis=1), target_names=classes)
        
        print("Classification Report:")
        print(report_str)

        report_dict = classification_report(test_labels, np.argmax(predictions, axis=1), target_names=classes, output_dict=True)
        f1_score = report_dict['macro avg']['f1-score']
        print(f"F1 Score: {f1_score}")

    def plot_confusion_matrix(self, test_data, model, class_names):
        predictions = model.predict(test_data)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_data.classes
        cm = confusion_matrix(true_classes, predicted_classes)

        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

if __name__ == "__main__":
    img_size = 256
    num_classes = 26
    batch_size = 64
    epochs = 30
    train_folder = r"train"
    test_folder = r"validation"

    malware_detection = MalwareDetection(img_size, num_classes, batch_size, epochs, train_folder, test_folder)
    malware_detection.train_model()
