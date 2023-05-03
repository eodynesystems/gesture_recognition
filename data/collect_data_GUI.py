import myo
import numpy as np
import time
import os
import tkinter as tk
import threading
from threading import Lock
import platform
from utils import EmgCollector, GIFViewer, gesture_display_text

if platform.system() == "Windows":
    sdk_path = 'C:\\myo-sdk-win-0.9.0\\'
elif platform.system() == "Darwin":
    sdk_path = os.path.abspath('SDK/myo-sdk-mac-0.9.0')


rec_duration = 4
gesture_pairs = {"Open-Close Hand":["OpenHand", "Neutral","ClosedHand"],
                "Thumb Abduction-Adduction":["ThumbAbduction", "Neutral", "ThumbAdduction"],
                "Wrist Pronation-Supination":["WristPronation", "Neutral", "WristSupination"]}
gif_viewer = None
start_button = None
gif_label = None
discard_button = None

def collect_emg_data(name, forearm_circumference, gesture):  
    global gesture_index  
    if gif_viewer is not None:
        gif_viewer.destroy()

    # Initialize MYO and start recording
    myo.init(sdk_path=sdk_path)
    hub = myo.Hub()

    gesture_index = 0
    while gesture_index < 5:
        if gesture in gesture_pairs:
            for idx, gest in enumerate(gesture_pairs[gesture]):
                countdown = 5 if idx == 0 else 3
                start = time.time()
                listener = EmgCollector(start, name, forearm_circumference, gest, gesture_index, root)
                listener.start_recording(countdown)
                hub.run(listener, rec_duration * 1000)
                listener.stop_recording()
        else:
            start = time.time()
            listener = EmgCollector(start, name, forearm_circumference, gesture, gesture_index, root)
            listener.start_recording()
            hub.run(listener, rec_duration * 1000)
            listener.stop_recording()
        
        gesture_index+=1
    reset_GUI(root)

def decrease_by_one(x):
    x=-1

def clean_gif(gif_viewer, gif_label, next_button):
    gif_viewer.pack_forget()
    gif_label.pack_forget()
    next_button.pack_forget()
    # Add start button
    start_button = tk.Button(root, text="Start", command=lambda: collect_emg_data(participant_entry.get(),
    float(forearm_entry.get()),
    gesture_var.get()))
    start_button.pack()  

    # add discard button
    discard_button = tk.Button(root, text="Discard last", command=lambda: decrease_by_one(gesture_index))
    discard_button.pack()

def show_gif(gesture, next_button, gif_label, pair=False):
    global gif_viewer
    gif_label.config(text = gesture_display_text[gesture])
    next_button.destroy()
    if gif_viewer is not None:
        gif_viewer.destroy()
    gif_viewer = GIFViewer(root, filename=f"media\gesture_demos\{gesture}.gif")
    gif_viewer.pack()
    gif_viewer.play()
    if not pair:
        _next_button = tk.Button(root, text="Next >>", command=lambda: clean_gif(gif_viewer, gif_label, _next_button))
        _next_button.pack()      

def show_tutorial(root, gesture):
    global gif_label
    pair = False
    gif_label = tk.Label(root, font=("Helvetica", 15), pady=40)
    gif_label.pack()
    if gesture in gesture_pairs:
        pair=True
        gestures = gesture_pairs[gesture].copy()
        gestures.pop(1)
    else:
        gestures = [gesture]
    show_gif(gestures[0], next_button, gif_label, pair)
    if pair:
        _next_button = tk.Button(root, text="Next >>", command=lambda: show_gif(gestures[1], _next_button, gif_label))
        _next_button.pack()

# Reset GUI
def reset_GUI(root):
    global next_button, start_button, gif_viewer, gif_label
    for widget in root.winfo_children():
        if (isinstance(widget, tk.Button)):
            widget.pack_forget()
    if gif_viewer is not None:
        gif_viewer.destroy()
    if gif_label is not None:
        gif_label.pack_forget()

    if start_button is not None:
        start_button.pack_forget()
    if discard_button is not None:
        discard_button.pack_forget()

    # Add next button
    next_button = tk.Button(root, text="Next >>", command=lambda: show_tutorial(root, gesture_var.get()))
    next_button.pack()

    # Add reset button
    reset_button = tk.Button(root, text="Reset", command=lambda: reset_GUI(root))
    reset_button.pack()


# Create GUI
root = tk.Tk()
root.title("EMG Data Collection")
root.geometry("400x600")

# Add participant name input
participant_label = tk.Label(root, text="Participant Name:")
participant_label.pack()
participant_entry = tk.Entry(root)
participant_entry.pack()

# Add forearm circumference input
forearm_label = tk.Label(root, text="Forearm Circumference (cm):")
forearm_label.pack()
forearm_entry = tk.Entry(root)
forearm_entry.insert(0, 0)
forearm_entry.pack()

# Add gesture dropdown menu
gesture_label = tk.Label(root, text="Gesture:")
gesture_label.pack()
gesture_var = tk.StringVar(root)
gesture_var.set("Select Gesture")
gesture_menu = tk.OptionMenu(root, gesture_var, "Open-Close Hand", "Thumb Abduction-Adduction", "Wrist Pronation-Supination", "Pinch", "Tripod", "Point", "Lateral")
gesture_menu.pack()

# Add next button
next_button = tk.Button(root, text="Next >>", command=lambda: show_tutorial(root, gesture_var.get()))
next_button.pack()

# Add reset button
reset_button = tk.Button(root, text="Reset", command=lambda: reset_GUI(root))
reset_button.pack()

#Run GUI 
root.mainloop()