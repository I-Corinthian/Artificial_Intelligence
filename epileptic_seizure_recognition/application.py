import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import model_runner


def load_data():
    global data, num_rows, current_row
    file_path = "epileptic_seizure_recognition/dataset/Epileptic_Seizure_Recognition.csv"
    data = pd.read_csv(file_path)
    num_rows = len(data)
    current_row = 0
    init_data()

def init_data():
    global current_row, line
    if current_row < num_rows:
        row_data = data.iloc[current_row].values
        y = row_data[1:-1] 
        truev = 1 if row_data[-1] == 1 else 0
        line.set_ydata(y)
        line.set_xdata(range(1, len(y) + 1))
        line.set_color(get_color(y))
        plt.title(f"DATA N.O {current_row + 1} TrueValue {truev}")
        canvas.draw()


def update_plot():
    global current_row, line
    if current_row < num_rows - 1:
        current_row += 1
        row_data = data.iloc[current_row].values
        y = row_data[1:-1] 
        truev = 1 if row_data[-1] == 1 else 0
        line.set_ydata(y)
        line.set_xdata(range(1, len(y) + 1))
        line.set_color(get_color(y))
        plt.title(f"DATA N.O {current_row + 1} TrueValue {truev}")
        canvas.draw()

def back_plot():
    global current_row, line
    if current_row > 0:
        current_row -= 1
        row_data = data.iloc[current_row].values
        y = row_data[1:-1] 
        truev = 1 if row_data[-1] == 1 else 0
        line.set_ydata(y)
        line.set_xdata(range(1, len(y) + 1))
        line.set_color(get_color(y))
        plt.title(f"DATA N.O {current_row + 1} TrueValue {truev}")
        canvas.draw()

def get_color(data):
    r = model_runner.prd(data)
    if r == 1:
        return 'red'
    else:
        return 'green'

root = tk.Tk()
root.title("Waveform Viewer")

frame = tk.Frame(root)
frame.pack()

load_button = tk.Button(frame, text="Load CSV", command=load_data)
load_button.pack(side=tk.LEFT)

back_button = tk.Button(frame, text="Back", command=back_plot)
back_button.pack(side=tk.LEFT)

next_button = tk.Button(frame, text="Next", command=update_plot)
next_button.pack(side=tk.LEFT)

fig, ax = plt.subplots(figsize=(10, 6))  
line, = ax.plot([], [], lw=2)
ax.set_xlim(0, 178) 
ax.set_ylim(-1600, 1600) 
ax.grid()

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()


root.mainloop()
