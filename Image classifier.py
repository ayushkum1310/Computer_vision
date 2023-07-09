import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from tensorflow.keras.applications import imagenet_utils, ConvNeXtXLarge
from tensorflow.keras.applications.resnet50 import decode_predictions
import cv2
import numpy as np

# Load the pre-trained model
model = ConvNeXtXLarge(weights='imagenet')

# Create a Tkinter window
window = tk.Tk()
window.title("Image Classifier")
window.configure(bg="white")
window.geometry("400x400")
window.configure(bg='#00FFFF')
# Function to classify an image
def classify_image():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                           filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))

    # Load and preprocess the image
    img = Image.open(file_path)
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = imagenet_utils.preprocess_input(img)

    # Make predictions
    predictions = predict(img)

    # Display the image and predicted class labels
    image_label.configure(image=ImageTk.PhotoImage(Image.open(file_path).resize((200, 200))))
    result_label.configure(text="Predicted Class:", fg="black", bg="lightblue", font=("Arial", 12, "bold"))
    prediction_listbox.delete(0, tk.END)
    for pred in predictions:
        prediction_listbox.insert(tk.END, f"{pred[1]}: {pred[2]*10:.2f}%")

    # Set foreground and background colors for the entire listbox
    prediction_listbox.config(fg="black", bg="white")


def predict(image):
    yhat = model.predict(image)
    decoded_predictions = decode_predictions(yhat, top=3)[0]
    return decoded_predictions


# Create GUI components
browse_button = tk.Button(window, text="Browse", command=classify_image, bg="lightblue", fg="black",
                          font=("Arial", 12, "bold"),borderwidth=0)
image_label = tk.Label(window, bg="#00FFFF")
result_label = tk.Label(window, bg="#00FFFF")

# Layout the GUI components
browse_button.pack()
image_label.pack()
result_label.pack()
prediction_listbox = tk.Listbox(window, height=5, width=30, selectbackground='lightblue', font=("Arial", 10))
prediction_listbox.pack()

# Start the Tkinter event loop
window.mainloop()
