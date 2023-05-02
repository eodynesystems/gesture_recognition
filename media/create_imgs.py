"""
This script extracts the last frames of all gifs in the directory and saves them as png files with the
same name 
"""
import os
from PIL import Image

directory = 'gesture_demos/'
def main():
    for filename in os.listdir(directory):
        if filename.endswith('.gif'):
            gif_path = os.path.join(directory, filename)
            with Image.open(gif_path) as im:
                # Get the last frame of the GIF
                im.seek(im.n_frames - 1)
                # Convert the frame to PNG and save it with the same filename as the GIF
                png_path = os.path.join(directory, os.path.splitext(filename)[0] + '.png')
                im.save(png_path, 'PNG')

if __name__ == "__main__":
    main()