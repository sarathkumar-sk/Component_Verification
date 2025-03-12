import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time

# Constants for side view calculations
AB_cm = 9

# Initialize cameras with reduced resolution for smoother processing
cap_top = cv2.VideoCapture(0)  # Top camera
cap_side = cv2.VideoCapture(2)  # Side camera
cap_top.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap_top.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap_side.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap_side.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Global frame storage
top_frame = None
side_frame = None
lock = threading.Lock()
run_flag = True

# Threaded frame capture
def capture_frames():
    global top_frame, side_frame, run_flag
    while run_flag:
        ret_top, frame_top = cap_top.read()
        ret_side, frame_side = cap_side.read()
        if ret_top and ret_side:
            with lock:
                top_frame = frame_top
                side_frame = frame_side
        time.sleep(0.01)

def classify_and_measure(contour, pixel_to_cm_ratio):
    if cv2.contourArea(contour) < 100:  # Ignore small contours
        return None, None

    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        width_cm = w / pixel_to_cm_ratio
        height_cm = h / pixel_to_cm_ratio
        return "Rectangle", (width_cm, height_cm)
    
    area = cv2.contourArea(contour)
    if area > 0:
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        if circularity > 0.75:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            diameter_cm = (2 * radius) / pixel_to_cm_ratio
            return "Circle", diameter_cm
    
    return None, None

def process_top_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return frame, thresh, []

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    pixel_to_cm_ratio = max(w, h) / 10.0

    cropped_frame = frame[y:y+h, x:x+w]
    cropped_thresh = thresh[y:y+h, x:x+w]
    shapes_within = []

    for contour in contours:
        if cv2.contourArea(contour) < cv2.contourArea(largest_contour):
            shape, dimensions = classify_and_measure(contour, pixel_to_cm_ratio)
            if shape:
                shapes_within.append((shape, dimensions))

    return cropped_frame, cropped_thresh, shapes_within

def process_side_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Find longest vertical line of black pixels (height detection)
    non_black_pixels = binary > 0
    vertical_lines = np.sum(non_black_pixels, axis=1)
    longest_line = max(vertical_lines) if np.any(vertical_lines) else 0

    if longest_line > 0:
        pixel_to_cm_ratio = AB_cm / longest_line
        object_height = longest_line * pixel_to_cm_ratio
    else:
        object_height = 0

    return binary, object_height

def update_gui():
    with lock:
        if top_frame is not None and side_frame is not None:
            # Process frames
            top_processed, top_segmented, shapes_within = process_top_frame(top_frame)
            side_segmented, object_height = process_side_frame(side_frame)

            # Top Frame
            top_frame_pil = Image.fromarray(cv2.cvtColor(top_processed, cv2.COLOR_BGR2RGB)).resize((300, 300))
            top_segmented_pil = Image.fromarray(top_segmented).resize((300, 300))

            top_frame_tk = ImageTk.PhotoImage(top_frame_pil)
            top_segmented_tk = ImageTk.PhotoImage(top_segmented_pil)

            label_top_frame.config(image=top_frame_tk)
            label_top_frame.image = top_frame_tk
            label_top_segmented.config(image=top_segmented_tk)
            label_top_segmented.image = top_segmented_tk

            # Side Frame
            side_frame_pil = Image.fromarray(cv2.cvtColor(side_frame, cv2.COLOR_BGR2RGB)).resize((300, 300))
            side_segmented_pil = Image.fromarray(side_segmented).resize((300, 300))

            side_frame_tk = ImageTk.PhotoImage(side_frame_pil)
            side_segmented_tk = ImageTk.PhotoImage(side_segmented_pil)

            label_side_frame.config(image=side_frame_tk)
            label_side_frame.image = side_frame_tk
            label_side_segmented.config(image=side_segmented_tk)
            label_side_segmented.image = side_segmented_tk

            # Display Shape Info
            if shapes_within:
                shapes_text = "\n".join(
                    [f"{s[0]}: {s[1][0]:.2f}cm x {s[1][1]:.2f}cm" if isinstance(s[1], tuple) 
                     else f"{s[0]}: Diameter {s[1]:.2f}cm" for s in shapes_within]
                )
                shapes_result.set(shapes_text)
            else:
                shapes_result.set("No shapes detected")

            height_result.set(f"Object Height: {object_height:.2f} cm")

    # Schedule next update
    window.after(30, update_gui)

# GUI Setup
window = tk.Tk()
window.title("Camera Calibration and Measurement")

# Frame Containers
left_panel = tk.Frame(window)
left_panel.pack(side="left")

right_panel = tk.Frame(window)
right_panel.pack(side="right")

# Top Frames
label_top_frame = tk.Label(left_panel)
label_top_frame.pack()
tk.Label(left_panel, text="Top Frame").pack()

label_top_segmented = tk.Label(left_panel)
label_top_segmented.pack()
tk.Label(left_panel, text="Top Segmented").pack()

# Side Frames
label_side_frame = tk.Label(right_panel)
label_side_frame.pack()
tk.Label(right_panel, text="Side Frame").pack()

label_side_segmented = tk.Label(right_panel)
label_side_segmented.pack()
tk.Label(right_panel, text="Side Segmented").pack()

# Results
shapes_result = tk.StringVar()
label_shapes = tk.Label(window, textvariable=shapes_result, font=("Helvetica", 12))
label_shapes.pack()

height_result = tk.StringVar()
label_height = tk.Label(window, textvariable=height_result, font=("Helvetica", 12))
label_height.pack()

# Start background thread for frame capture
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

# Start live update
update_gui()

# Start Tkinter event loop
window.mainloop()

# Cleanup
run_flag = False
capture_thread.join()
cap_top.release()
cap_side.release()
cv2.destroyAllWindows()

