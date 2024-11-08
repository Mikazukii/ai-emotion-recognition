import tkinter as tk
from tkinter import filedialog, messagebox, Button
from PIL import Image, ImageOps
import numpy as np
import cv2  # Used to capture image from webcam
from keras import load_model

model = load_model("models/augmented_facial_recognition.h5")


def preprocess_image(image):
    """Converts the image to grayscale, resizes to 48x48,
    and normalizes to (1, 48, 48, 1) with values between 0 and 1."""
    image_bw = ImageOps.grayscale(image)  # Convert to grayscale
    image_resized = image_bw.resize((48, 48))  # Resize to 48x48 pixels
    image_array = (
        np.array(image_resized).astype("float32") / 255.0
    )  # Normalize between 0 and 1
    image_array = image_array.reshape((1, 48, 48, 1))  # Reshape to (1, 48, 48, 1)
    return image_array


def take_picture():
    """Takes a photo with the webcam and returns it without saving."""
    cap = cv2.VideoCapture(0)  # Open the webcam
    ret, frame = cap.read()  # Capture a single frame
    cap.release()  # Release the webcam
    if ret:
        image = Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )  # Convert to PIL Image
        return image
    else:
        print("Failed to capture image.")
        return None


def import_image():
    """Opens a dialog to select an image and returns it."""
    root = tk.Tk()
    root.withdraw()
    image_file = filedialog.askopenfilename(
        title="Choose an image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")],
    )
    if image_file:
        image = Image.open(image_file)
        return image
    else:
        print("No image selected.")
        return None


def run_prediction(action):
    """Process the selected action, capture/import the image, and make a prediction."""
    if action == "picture":
        image = take_picture()
    elif action == "import":
        image = import_image()
    else:
        messagebox.showerror("Error", "Invalid action.")
        return

    if image:
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        messagebox.showinfo("Prediction Result", f"Prediction result: {prediction}")


def main():
    """Create the tkinter interface to choose between actions."""
    # Initialize the main window
    root = tk.Tk()
    root.title("Choose an Option")

    # Create a label to prompt for choice
    label = tk.Label(root, text="Please choose an option:")
    label.pack(pady=10)

    # Create buttons for the two choices
    button_picture = tk.Button(
        root, text="Take a Picture", command=lambda: run_prediction("picture")
    )
    button_picture.pack(pady=5)

    button_import = tk.Button(
        root, text="Import an Image", command=lambda: run_prediction("import")
    )
    button_import.pack(pady=5)

    # Start the Tkinter main loop
    root.mainloop()