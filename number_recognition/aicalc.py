import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
import tensorflow as tf
import cv2 as cv
import numpy as np
import os

model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), "models", "handwrittennum.keras"))

def predict_number(image_path):
    img = cv.imread(image_path)[:,:,0]
    img = np.invert(np.array([img]))
    img = img / 255.0 
    
    predictions = model.predict(img)
    predicted_number = np.argmax(predictions)
    
    return predicted_number

def on_drop1(event):
    global num1
    file_path = event.data
    num1 = predict_number(file_path)
    label1.config(text=f"Number 1: {num1}")

def on_drop2(event):
    global num2
    file_path = event.data
    num2 = predict_number(file_path)
    label2.config(text=f"Number 2: {num2}")

def calculate():
    if num1 is None or num2 is None:
        print("Please drop images in both boxes.")
        return

    operation = selected_operation.get()
    result = None

    if operation == "+":
        result = num1 + num2
    elif operation == "-":
        result = num1 - num2
    elif operation == "*":
        result = num1 * num2
    elif operation == "/":
        if num2 != 0:
            result = num1 / num2
        else:
            result = "undefined (division by zero)"

    print(f"Result: {result}")

num1 = None
num2 = None

root = TkinterDnD.Tk()
root.title("AI CALC")
root.geometry("600x400")

label1 = tk.Label(root, text="drop an image", bg="lightgray", width=30, height=10)
label1.grid(row=0, column=0, padx=20, pady=20)

label2 = tk.Label(root, text="drop an image", bg="lightgray", width=30, height=10)
label2.grid(row=0, column=1, padx=20, pady=20)

label1.drop_target_register(DND_FILES)
label1.dnd_bind('<<Drop>>', on_drop1)

label2.drop_target_register(DND_FILES)
label2.dnd_bind('<<Drop>>', on_drop2)

selected_operation = tk.StringVar(value="+")
operations = ["+", "-", "*", "/"]

for i, operation in enumerate(operations):
    rb = tk.Radiobutton(root, text=operation, variable=selected_operation, value=operation)
    rb.grid(row=i+1, column=0, padx=5, pady=5)

submit_button = tk.Button(root, text="Submit", command=calculate)
submit_button.grid(row=2, columnspan=4, pady=20)

root.mainloop()