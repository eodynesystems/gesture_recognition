import myo
import numpy as np
import time
import os
import tkinter as tk
import threading
from threading import Lock
from PIL import Image, ImageTk

gesture_display_text = {"OpenHand": "Open Hand", 
                        "ClosedHand": "Closed Hand",
                        "ThumbAbduction": "Thumb-in",
                        "ThumbAdduction": "Thumb-out",
                        "WristPronation": "Palm-down",
                        "WristSupination": "Palm-up",
                        "Pinch": "Pinch",
                        "Tripod":"Tripod",
                        "Point":"Point",
                        "Lateral":"Lateral",
                        "Neutral":"Rest"}

class GIFViewer(tk.Label):
    def __init__(self, master=None, filename=None, **kw):
        self._gif_frames = []
        tk.Label.__init__(self, master, **kw)
        if filename:
            self.load(filename)

    def load(self, filename):
        with Image.open(filename) as im:
            for frame in range(0, im.n_frames):
                im.seek(frame)
                frame_image = ImageTk.PhotoImage(im)
                self._gif_frames.append(frame_image)

    def play(self):
        animate_thread = threading.Thread(target=self._animate_gif, args = [0])
        animate_thread.start()

    def _animate_gif(self, frame):
        self.config(image=self._gif_frames[frame])
        frame = (frame + 1) % len(self._gif_frames)
        self.after(40, self._animate_gif, frame)


class EmgCollector(myo.DeviceListener):
    def __init__(self, start, name, forearm_circumference, gesture, gesture_index, root):
        self.name = name
        self.start = start
        self.forearm_circumference = forearm_circumference
        self.gesture = gesture
        self.gesture_index = gesture_index
        self.root = root
        self.emg_data = np.zeros((1, 8))
        self.lock = Lock()
        self.is_recording = False
        self.countdown_label = tk.Label(root, font=("Helvetica", 15), pady=40)
        self.countdown_label.pack()
        self.reference_img_label = None

    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_emg(self, event):
        if self.is_recording:
            self.emg_data = np.concatenate((self.emg_data, np.reshape(np.array(event.emg), (1, 8))), axis=0)

    
    def start_recording(self, countdown_secs = 5):
        # Start countdown timer
        for i in range(countdown_secs):
            self.countdown_label.config(text=f"Recording {gesture_display_text[self.gesture]} in {countdown_secs-i} seconds...")
            self.root.update()
            time.sleep(1)

        # Load and show reference gesture GIF
        reference_img_filename = f"media/gesture_demos/{self.gesture}.png"
        reference_img = Image.open(reference_img_filename)
        reference_img = reference_img.resize((200, 200), Image.ANTIALIAS)
        reference_img = ImageTk.PhotoImage(reference_img)
        self.reference_img_label = tk.Label(self.root, image=reference_img)
        self.reference_img_label.image = reference_img
        self.reference_img_label.pack()

        # Start recording
        self.is_recording = True
        self.emg_data = np.zeros((1, 8))
        self.countdown_label.config(text=f"Recording {gesture_display_text[self.gesture]}...")
        self.root.update()

    def stop_recording(self):
        # Remove reference gif
        if self.reference_img_label is not None:
            self.reference_img_label.destroy()

        # Stop recording
        self.is_recording = False
        self.countdown_label.config(text=f"Saving {gesture_display_text[self.gesture]} data...")
        self.root.update()

        # Save EMG data to file
        if not os.path.isdir(f"data/{self.name}"):
            os.mkdir(f"data/{self.name}")

        emg_data_array = np.array(self.emg_data)
        filename = f"data/{self.name}/{self.gesture}_{self.gesture_index}.npy"
        i=1
        while os.path.exists(filename):
            filename = f"data/{self.name}/{self.gesture}_{self.gesture_index + i}.npy"
            i+=1

        np.save(filename, emg_data_array)
        
        # Save forearm size 
        filename = f"data/{self.name}/forearm_circumference.npy"
        np.save(filename, np.array(self.forearm_circumference))
        
        # Increment gesture index
        self.countdown_label.destroy()
        self.root.update()

        # Reset EMG
        self.emg_data = np.zeros((1, 8))
