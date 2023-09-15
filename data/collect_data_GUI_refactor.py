import tkinter as tk
from tkinter import ttk
import time

# Global variables
participant_name = ""
selected_gesture = ""
recording_count = 0
recording_started = False

# Dummy function to simulate recording
def start_recording():
    global recording_started
    recording_started = True
    print("Recording started")

# Dummy function to simulate stopping recording
def stop_recording():
    global recording_started
    recording_started = False
    print("Recording stopped")

# Dummy function to simulate discarding the last recording
def discard_last_recording():
    print("Last recording discarded")

# Dummy function to simulate resetting the UI
def reset_ui():
    global recording_count
    recording_count = 0
    print("UI reset")

# Function to handle the continue button press
def continue_button_pressed():
    global next_button
    global participant_name, selected_gesture
    participant_name = participant_name_entry.get()
    selected_gesture = gesture_var.get()

    next_button = ttk.Button(root, text="Next", command=next_button_pressed)
    next_button.pack()

# Function to handle the next button press
def next_button_pressed():
    global recording_count, start_button

    # Hide the next button
    next_button.pack_forget()

    # Play GIF or perform any other action here
    # Simulating with sleep for 3 seconds
    time.sleep(3)

    # Start recording
    start_button = ttk.Button(root, text="Start", command=start_recording)
    start_button.pack()

# Function to handle the start button press
def start_button_pressed():
    global recording_count, discard_button

    # Disable start button and show discard button
    start_button.configure(state="disabled")
    discard_button.pack()

    # Perform countdown and recording
    countdown_label = tk.Label(root, text="Countdown: 5")
    countdown_label.pack()

    for i in range(5, 0, -1):
        countdown_label.configure(text="Countdown: " + str(i))
        root.update()
        time.sleep(1)

    countdown_label.configure(text="Recording...")
    root.update()
    time.sleep(3)  # Recording for 3 seconds

    # Stop recording and hide discard button
    stop_recording()
    discard_button.pack_forget()

    # Rest period
    time.sleep(3)  # Rest for 3 seconds

    # Increment recording count
    recording_count += 1

    # Check if the desired number of recordings is reached
    if recording_count == 4:
        # Reset UI
        reset_ui()
    else:
        # Repeat the recording process
        start_button_pressed()

# Function to handle the discard button press
def discard_button_pressed():
    discard_last_recording()

    # Start the recording process again
    start_button_pressed()


# Create the main Tkinter window
root = tk.Tk()
root.title("EMG Data Collection")

# Participant name entry field
participant_name_label = tk.Label(root, text="Participant Name:")
participant_name_label.pack()
participant_name_entry = ttk.Entry(root)
participant_name_entry.pack()

# Gesture selection drop-down menu
gesture_label = tk.Label(root, text="Select Gesture:")
gesture_label.pack()
gesture_var = tk.StringVar(root)
gesture_dropdown = ttk.Combobox(root, textvariable=gesture_var, values=["Open", "Close", "Rest"])
gesture_dropdown.pack()

# Continue button
continue_button = ttk.Button(root, text="Continue", command=continue_button_pressed)
continue_button.pack()
# Discard button
discard_button = ttk.Button(root, text="Discard", command=discard_button_pressed)

# Run the Tkinter event loop
root.mainloop()

