import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import time
import pygame  

pygame.mixer.init()

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Build the model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Class for the digit recognition app
class DigitRecognizerApp:
    def __init__(self, model):  # Correctly defined constructor
        self.model = model
        self.window = tk.Tk()
        self.window.title("Handwritten Digit Recognition")
        self.window.config(bg="#d1e7f5")

        # Play Christmas song on app launch
        pygame.mixer.music.load("christmas_song.mp3")  # Path to the MP3 file
        pygame.mixer.music.play(-1)  # Play the song in a loop

        self.title_label = tk.Label(self.window, text="🎄 Handwritten Digit Recognition 🎄", font=("Apple Chancery", 24, "bold"), bg="#d1e7f5")
        self.title_label.pack(pady=10)
        
        self.desc_label = tk.Label(self.window, text="Desenează o cifră și apasă 'Predict' pentru a afla rezultatul.", font=("Apple Chancery", 14), bg="#d1e7f5")
        self.desc_label.pack(pady=5)
        
        self.canvas = tk.Canvas(self.window, width=200, height=200, bg="white", borderwidth=2, relief="ridge")
        self.canvas.pack(pady=10)
        
        self.predict_button = tk.Button(self.window, text="Predict", font=("Apple Chancery", 14), command=self.predict, bg="#4caf50", fg="white", activebackground="#388e3c", width=15)
        self.predict_button.pack(pady=5)
        
        self.clear_button = tk.Button(self.window, text="Clear", font=("Apple Chancery", 14), command=self.clear, bg="#f44336", fg="white", activebackground="#d32f2f", width=15)
        self.clear_button.pack(pady=5)
        
        self.result_label = tk.Label(self.window, text="", font=("Apple Chancery", 18, "bold"), bg="#d1e7f5")
        self.result_label.pack(pady=10)
        
        self.animation_label = tk.Label(self.window, text="", font=("Apple Chancery", 18, "bold"), bg="#d1e7f5", fg="red")
        self.animation_label.pack(pady=10)

        self.canvas.bind("<B1-Motion>", self.draw)
        self.image = Image.new("L", (200, 200), 255)
        self.draw_image = ImageDraw.Draw(self.image)
        
        self.window.mainloop()
    
    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-8, y-8, x+8, y+8, fill="black", width=8)
        self.draw_image.ellipse([x-8, y-8, x+8, y+8], fill="black")
    
    def clear(self):
        self.canvas.delete("all")
        self.draw_image.rectangle((0, 0, 200, 200), fill="white")
        self.result_label.config(text="")
        self.animation_label.config(text="")
    
    def predict(self):
        img = self.image.resize((28, 28)).convert("L")
        img = ImageOps.invert(img)
        img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0
        prediction = self.model.predict(img_array)
        digit = np.argmax(prediction)
        self.result_label.config(text=f"Digit: {digit}", fg="#ff6347")

        # Show "Crăciun Fericit!" animation
        self.animate_christmas()

    def animate_christmas(self):
        for _ in range(3):
            self.animation_label.config(text="Crăciun Fericit!", fg="red")
            self.window.update()
            time.sleep(0.5)
            self.animation_label.config(text="")
            self.window.update()
            time.sleep(0.5)
        self.animation_label.config(text="Crăciun Fericit!", fg="red")

# Run the application
app = DigitRecognizerApp(model)
