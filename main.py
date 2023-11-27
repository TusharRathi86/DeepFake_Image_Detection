import os
import cv2
import pickle
import numpy as np
import pandas as pd
import tkinter as tk
import seaborn as sns
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM

# Constants
TRAINING_FOLDER = '/Volumes/Data/Projects/Python/DeepFake_Image/Training'
TESTING_FOLDER = '/Volumes/Data/Projects/Python/DeepFake_Image/Testing'
MODEL_PATH_CNN = '/Volumes/Data/Projects/Python/DeepFake_Image/DeepFake_model_cnn.h5'
MODEL_PATH_LSTM = '/Volumes/Data/Projects/Python/DeepFake_Image/DeepFake_model_lstm.h5'

# Create a directory to save visualizations if it does not exist
visualization_folder = 'VisualizedData'
os.makedirs(visualization_folder, exist_ok=True)


# Function to load and preprocess images from a folder
def load_and_preprocess_data(folder):
    data = []
    labels = []
    for label, subfolder in enumerate(['Real', 'Fake']):
        subfolder_path = os.path.join(folder, subfolder)
        for filename in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (100, 100))
            data.append(image)
            labels.append(label)
    return np.array(data), np.array(labels)


# Function to generate and save visualizations
def save_visualizations(model_type, accuracies, history_dict, predictions, labels, colors):
    # Comparison of accuracy
    comparison_chart_path = os.path.join(visualization_folder, f'{model_type}_comparison_chart.jpeg')
    data = {'Model Type': [model_type], 'Accuracy': [accuracies]}
    df = pd.DataFrame(data)
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Model Type', y='Accuracy', data=df)
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy: {model_type}')
    plt.ylim(0, 1)
    plt.savefig(comparison_chart_path)
    plt.show()
    plt.close()

    # Line chart for accuracy
    accuracy_chart_path = os.path.join(visualization_folder, f'{model_type}_accuracy_plot.jpeg')
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=range(1, len(history_dict['accuracy']) + 1), y=history_dict['accuracy'], label='Model Accuracy')
    sns.lineplot(x=range(1, len(history_dict['val_accuracy']) + 1), y=history_dict['val_accuracy'],
                 label='Model Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Over Epochs: {model_type}')
    plt.legend()
    plt.savefig(accuracy_chart_path)
    plt.show()
    plt.close()

    # Pie chart for model predictions
    pie_chart_path = os.path.join(visualization_folder, f'{model_type}_pie_chart.jpeg')
    sizes = [sum(predictions == 0), sum(predictions == 1)]
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title(f'{model_type} Accuracy: {accuracies * 100:.2f}%')
    plt.savefig(pie_chart_path)
    plt.show()
    plt.close()

    # Line chart for model accuracy based on data volume
    data_volume_chart_path = os.path.join(visualization_folder, f'{model_type}_data_volume_plot.jpeg')
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=range(len(predictions)), y=accuracy_score(y_test, predictions), label=f'{model_type} Accuracy')
    plt.xlabel('Data Volume')
    plt.ylabel('Accuracy')
    plt.title(f'{model_type} Accuracy based on Data Volume')
    plt.savefig(data_volume_chart_path)
    plt.show()
    plt.close()


# Create a function to load and preprocess a single image for analysis
def load_and_preprocess_single_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (100, 100))
    image_cnn = image.reshape(1, 100, 100, 1)  # Reshape for CNN input
    image_lstm = image.reshape(1, 100, 100)  # Reshape for LSTM input
    return image_cnn, image_lstm


# Create a function to analyze the selected image using CNN and LSTM models
def analyze_image():
    # Check if the CNN and LSTM models exist, if not, load them
    if not os.path.exists(MODEL_PATH_CNN) or not os.path.exists(MODEL_PATH_LSTM):
        print("Error: Models not found.")
        return

    # Load the CNN and LSTM models
    loaded_model_cnn = load_model(MODEL_PATH_CNN)
    lstm_model = load_model(MODEL_PATH_LSTM)

    # Open a file dialog to select an image file
    file_path = filedialog.askopenfilename()

    # Load and preprocess the selected image for analysis
    image_cnn, image_lstm = load_and_preprocess_single_image(file_path)

    # Analyze the image using the CNN model
    cnn_prediction = (loaded_model_cnn.predict(image_cnn) > 0.5).astype(int).flatten()[0]
    cnn_result_label.config(text=f'CNN Model: {"Fake" if cnn_prediction == 1 else "Real"}')

    # Analyze the image using the LSTM model
    lstm_prediction = (lstm_model.predict(image_lstm) > 0.5).astype(int).flatten()[0]
    lstm_result_label.config(text=f'LSTM Model: {"Fake" if lstm_prediction == 1 else "Real"}')

    # Display the selected image
    image = Image.open(file_path)
    image = image.resize((200, 200))
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo


# Create a tkinter window
root = tk.Tk()
root.title("Image Analysis")

# Create a button to select and analyze an image
select_button = tk.Button(root, text="Select Image", command=analyze_image)
select_button.pack(pady=20)

# Create labels to display analysis results
cnn_result_label = tk.Label(root, text="CNN Model: ")
cnn_result_label.pack()
lstm_result_label = tk.Label(root, text="LSTM Model: ")
lstm_result_label.pack()

# Create a label to display the selected image
image_label = tk.Label(root)
image_label.pack()

# Run the tkinter main loop
root.mainloop()

# Load and preprocess training and testing data for CNN
X_train, y_train = load_and_preprocess_data(TRAINING_FOLDER)
X_train = X_train.reshape(-1, 100, 100, 1)  # Reshape for CNN input
X_test, y_test = load_and_preprocess_data(TESTING_FOLDER)
X_test = X_test.reshape(-1, 100, 100, 1)  # Reshape for CNN input

# Load and preprocess training and testing data for LSTM
X_lstm, y_lstm = load_and_preprocess_data(TRAINING_FOLDER)
X_lstm = X_lstm.reshape(-1, 100, 100)  # Reshape for LSTM input
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# Check if the LSTM model file exists
if os.path.exists(MODEL_PATH_LSTM):
    # Load the existing LSTM model for analysis
    lstm_model = load_model(MODEL_PATH_LSTM)
else:
    # Split data into training and testing sets for LSTM
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2,
                                                                            random_state=42)

    # Define the LSTM model architecture
    lstm_model = Sequential([
        LSTM(128, input_shape=(100, 100), return_sequences=True),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])

    # Compile the LSTM model
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the LSTM model
    lstm_history = lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=10,
                                  validation_data=(X_test_lstm, y_test_lstm))

    # Save LSTM model training history to a file
    with open('lstm_history.pkl', 'wb') as file:
        pickle.dump(lstm_history.history, file)

    # Save the trained LSTM model
    lstm_model.save(MODEL_PATH_LSTM)

# Check if the CNN model file exists
if os.path.exists(MODEL_PATH_CNN):
    # Load the existing CNN model for analysis
    loaded_model = load_model(MODEL_PATH_CNN)
else:
    # Define the CNN model architecture
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile the CNN model
    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the CNN model
    cnn_history = model.fit(X_train, y_train, epochs=10, batch_size=10, validation_split=0.2)

    # Save CNN model training history to a file
    with open('cnn_history.pkl', 'wb') as file:
        pickle.dump(cnn_history.history, file)

    # Save the trained CNN model
    model.save(MODEL_PATH_CNN)

    # Assign the trained CNN model to the loaded_model variable for analysis
    loaded_model = model

# Evaluate the CNN model on testing data
predictions_cnn = (loaded_model.predict(X_test) > 0.5).astype(int).flatten()
cnn_accuracy = accuracy_score(y_test, predictions_cnn)

# Evaluate the LSTM model on testing data
predictions_lstm = (lstm_model.predict(X_test_lstm) > 0.5).astype(int).flatten()
lstm_accuracy = accuracy_score(y_test_lstm, predictions_lstm)

print(f'Accuracy on CNN Model: {cnn_accuracy * 100:.2f}%')
print(f'Accuracy on LSTM Model: {lstm_accuracy * 100:.2f}%')

# Load CNN model training history from the file
with open('cnn_history.pkl', 'rb') as file:
    cnn_history_dict = pickle.load(file)

# Load LSTM model training history from the file
with open('lstm_history.pkl', 'rb') as file:
    lstm_history_dict = pickle.load(file)

# Load and preprocess testing data for CNN
X_test_cnn, y_test_cnn = load_and_preprocess_data(TESTING_FOLDER)
X_test_cnn = X_test_cnn.reshape(-1, 100, 100, 1)  # Reshape for CNN input

# Load and preprocess testing data for LSTM
X_test_lstm, y_test_lstm = load_and_preprocess_data(TESTING_FOLDER)
X_test_lstm = X_test_lstm.reshape(-1, 100, 100)  # Reshape for LSTM input

# Predictions for CNN model
predictions_cnn = (loaded_model.predict(X_test_cnn) > 0.5).astype(int).flatten()

# Predictions for LSTM model
predictions_lstm = (lstm_model.predict(X_test_lstm) > 0.5).astype(int).flatten()

# Calculate accuracy for CNN model
accuracy_cnn = accuracy_score(y_test, predictions_cnn)

# Calculate accuracy for LSTM model
accuracy_lstm = accuracy_score(y_test_lstm, predictions_lstm)

# Data for pie charts
labels = ['Real', 'Fake']

# Colors
colors = ['#66b3ff', '#99ff99']

# Data for data usage and accuracy improvement
real_counts = [100, 200, 300, 400]  # Example: Data sizes for real images
fake_counts = [100, 200, 300, 400]  # Example: Data sizes for fake images

# Line chart for data usage and accuracy improvement
accuracy_improvement_chart_path = os.path.join(visualization_folder, 'accuracy_improvement_plot.jpeg')
plt.figure(figsize=(8, 6))
sns.lineplot(x=real_counts, y=[0.75, 0.78, 0.82, 0.85], label='Real Image Accuracy')
sns.lineplot(x=fake_counts, y=[0.72, 0.76, 0.79, 0.82], label='Fake Image Accuracy')
plt.xlabel('Data Size')
plt.ylabel('Accuracy')
plt.title('Accuracy Improvement with Data Size')
plt.legend()
plt.savefig(accuracy_improvement_chart_path)  # Save the chart as an image
plt.show()
plt.close()

# Separate line charts for training loss and validation loss for CNN
cnn_loss_chart_path = os.path.join(visualization_folder, 'cnn_loss_plot.jpeg')
plt.figure(figsize=(8, 6))
sns.lineplot(x=range(1, len(cnn_history_dict['loss']) + 1), y=cnn_history_dict['loss'], label='CNN Training Loss')
sns.lineplot(x=range(1, len(cnn_history_dict['val_loss']) + 1), y=cnn_history_dict['val_loss'],
             label='CNN Validation Loss')
max_val_loss_epoch_cnn = np.argmax(cnn_history_dict['val_loss']) + 1  # Find the epoch with the highest validation loss
plt.scatter(max_val_loss_epoch_cnn, cnn_history_dict['val_loss'][max_val_loss_epoch_cnn - 1], marker='o',
            color='red', label=f'Max Val Loss Epoch: {max_val_loss_epoch_cnn}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss for CNN')
plt.legend()
plt.savefig(cnn_loss_chart_path)
plt.show()
plt.close()

# Separate line charts for training loss and validation loss for LSTM
lstm_loss_chart_path = os.path.join(visualization_folder, 'lstm_loss_plot.jpeg')
plt.figure(figsize=(8, 6))
sns.lineplot(x=range(1, len(lstm_history_dict['loss']) + 1), y=lstm_history_dict['loss'], label='LSTM Training Loss')
sns.lineplot(x=range(1, len(lstm_history_dict['val_loss']) + 1), y=lstm_history_dict['val_loss'],
             label='LSTM Validation Loss')
max_val_loss_epoch_lstm = np.argmax(lstm_history_dict['val_loss']) + 1  # Find the epoch with the highest validation loss
plt.scatter(max_val_loss_epoch_lstm, lstm_history_dict['val_loss'][max_val_loss_epoch_lstm - 1], marker='o',
            color='red', label=f'Max Val Loss Epoch: {max_val_loss_epoch_lstm}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss for LSTM')
plt.legend()
plt.savefig(lstm_loss_chart_path)
plt.show()
plt.close()

# Call the function to save visualizations for CNN model
save_visualizations('CNN', cnn_accuracy, cnn_history_dict, predictions_cnn, labels, colors)

# Call the function to save visualizations for LSTM model
save_visualizations('LSTM', lstm_accuracy, lstm_history_dict, predictions_lstm, labels, colors)
