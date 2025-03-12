import cv2
import numpy as np
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

# Constants
AB_cm = 8   # cm
OC_cm = 10  # cm
BC_cm = 20  # cm

def calculate_object_height(image, AB_cm):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thresholding to binary image to find AB in pixels
    _, binary_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Find the longest vertical line of black pixels (AB)
    ab_pixels = find_longest_contiguous_black_line(binary_image)

    # Convert image to HSV format
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color range for the object (example range for orange color)
    lower_bound = np.array([5, 150, 150])  # Adjust these values based on the object's color
    upper_bound = np.array([15, 255, 255])
    
    # Create a mask for the object
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    
    # Mask out white floor (assuming floor is close to white)
    white_floor_mask = cv2.inRange(gray_image, 200, 255)
    mask = cv2.bitwise_and(mask, mask, mask=cv2.bitwise_not(white_floor_mask))

    # Find the longest vertical line of non-black pixels (object height)
    obj_height_pixels = find_longest_contiguous_non_black_line(mask)

    # Calculate pixel to cm ratio
    pixel_to_cm_ratio = AB_cm / ab_pixels

    # Calculate the height of the object (OX)
    height_cm = obj_height_pixels * pixel_to_cm_ratio

    return height_cm, mask

def find_longest_contiguous_black_line(binary_image):
    # Convert binary image to black and white
    black_pixels = binary_image == 0

    # Find vertical lines of black pixels
    vertical_lines = np.sum(black_pixels, axis=1)
    longest_black_line = max(vertical_lines) if np.any(vertical_lines) else 0
    
    return longest_black_line

def find_longest_contiguous_non_black_line(binary_image):
    # Convert binary image to black and white
    non_black_pixels = binary_image > 0

    # Find vertical lines of non-black pixels
    vertical_lines = np.sum(non_black_pixels, axis=1)
    longest_non_black_line = max(vertical_lines) if np.any(vertical_lines) else 0
    
    return longest_non_black_line

def update_frame():
    ret, frame = cap.read()
    if ret:
        # Convert image to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        img = Image.fromarray(rgb_frame)
        img = ImageTk.PhotoImage(img)
        # Update label
        lbl_video.imgtk = img
        lbl_video.configure(image=img)
    lbl_video.after(10, update_frame)

def capture_and_calculate():
    ret, frame = cap.read()
    if ret:
        height_cm, thresholded_image = calculate_object_height(frame, AB_cm)
        
        # Convert thresholded image to RGB for display
        thresholded_image_rgb = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2RGB)
        img_threshold = Image.fromarray(thresholded_image_rgb)
        img_threshold = ImageTk.PhotoImage(img_threshold)
        
        # Update the thresholded image label
        lbl_thresholded.imgtk = img_threshold
        lbl_thresholded.configure(image=img_threshold)
        
        # Display the calculated height
        lbl_result.config(text=f"Object Height: {height_cm:.2f} cm")

# Setup the main window
root = tk.Tk()
root.title("Camera Feed")

# Video capture
cap = cv2.VideoCapture(0)  # Changed parameter to 1

# Create labels for video feed and thresholded image
frame_width = 640
frame_height = 480
lbl_video = Label(root)
lbl_video.grid(row=0, column=0)
lbl_thresholded = Label(root)
lbl_thresholded.grid(row=0, column=1)

# Create a capture button
btn_capture = tk.Button(root, text="Capture", command=capture_and_calculate)
btn_capture.grid(row=1, column=0, columnspan=2)

# Create a label for the result
lbl_result = Label(root, text="Object Height: N/A")
lbl_result.grid(row=2, column=0, columnspan=2)

# Start updating the video feed
update_frame()

# Run the Tkinter event loop
root.mainloop()

# Release the capture when done
cap.release()
cv2.destroyAllWindows()
