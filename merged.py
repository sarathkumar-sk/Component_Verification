import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Constants for the side view calculations
AB_cm = 9
OC_cm = 10
BC_cm = 22

# Initialize cameras
cap_top = cv2.VideoCapture(0)  # Top camera
cap_side = cv2.VideoCapture(2)  # Side camera

def classify_and_measure(contour, pixel_to_cm_ratio):
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        width_cm = w / pixel_to_cm_ratio
        height_cm = h / pixel_to_cm_ratio
        return "Rectangle", (width_cm, height_cm)
    else:
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
    contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    pixel_size_of_box = max(w, h)
    pixel_to_cm_ratio = pixel_size_of_box / 10.0
    cropped_frame = frame[y:y+h, x:x+w]
    cropped_thresh = thresh[y:y+h, x:x+w]
    cropped_contours, _ = cv2.findContours(cropped_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    shapes_within = []
    largest_shape, largest_dimensions = None, None
    if cropped_contours:
        valid_contours = [c for c in cropped_contours if cv2.contourArea(c) < cv2.contourArea(largest_contour)]
        if valid_contours:
            largest_cropped_contour = max(valid_contours, key=cv2.contourArea)
            largest_shape, largest_dimensions = classify_and_measure(largest_cropped_contour, pixel_to_cm_ratio)
            cv2.drawContours(cropped_frame, [largest_cropped_contour], -1, (0, 255, 0), 2)
            for contour in valid_contours:
                if contour is not largest_cropped_contour:
                    shape, dimensions = classify_and_measure(contour, pixel_to_cm_ratio)
                    if shape and dimensions:
                        shapes_within.append((shape, dimensions))
    return cropped_frame, cropped_thresh, largest_shape, largest_dimensions, shapes_within, pixel_to_cm_ratio, x, y, w, h

def calculate_object_distance_from_box_bottom(cropped_thresh, pixel_to_cm_ratio, box_bottom_y):
    non_black_pixels = cropped_thresh > 0
    vertical_lines = np.sum(non_black_pixels, axis=1)
    object_distance_pixels = np.argmax(vertical_lines)
    object_distance_cm = object_distance_pixels / pixel_to_cm_ratio
    distance_from_box_bottom_cm = box_bottom_y / pixel_to_cm_ratio
    return distance_from_box_bottom_cm

def calculate_object_height(image, AB_cm, additional_distance_cm):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
    ab_pixels = find_longest_contiguous_black_line(binary_image)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([5, 150, 150])
    upper_bound = np.array([15, 255, 255])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    white_floor_mask = cv2.inRange(gray_image, 200, 255)
    mask = cv2.bitwise_and(mask, mask, mask=cv2.bitwise_not(white_floor_mask))
    obj_height_pixels = find_longest_contiguous_non_black_line(mask)
    pixel_to_cm_ratio = AB_cm / ab_pixels
    height_cm = (obj_height_pixels * pixel_to_cm_ratio) + additional_distance_cm
    return height_cm, mask

def find_longest_contiguous_black_line(binary_image):
    black_pixels = binary_image == 0
    vertical_lines = np.sum(black_pixels, axis=1)
    longest_black_line = max(vertical_lines) if np.any(vertical_lines) else 0
    return longest_black_line

def find_longest_contiguous_non_black_line(binary_image):
    non_black_pixels = binary_image > 0
    vertical_lines = np.sum(non_black_pixels, axis=1)
    longest_non_black_line = max(vertical_lines) if np.any(vertical_lines) else 0
    return longest_non_black_line

def update_gui(top_frame, top_segmented, top_shape, top_dimensions, top_shapes_within, side_frame, side_segmented, side_height):
    top_frame_pil = Image.fromarray(cv2.cvtColor(top_frame, cv2.COLOR_BGR2RGB))
    top_segmented_pil = Image.fromarray(top_segmented)
    side_frame_pil = Image.fromarray(cv2.cvtColor(side_frame, cv2.COLOR_BGR2RGB))
    side_segmented_pil = Image.fromarray(side_segmented)
    
    top_frame_resized = top_frame_pil.resize((300, 300))
    top_segmented_resized = top_segmented_pil.resize((300, 300))
    side_frame_resized = side_frame_pil.resize((300, 300))
    side_segmented_resized = side_segmented_pil.resize((300, 300))
    
    top_frame_tk = ImageTk.PhotoImage(top_frame_resized)
    top_segmented_tk = ImageTk.PhotoImage(top_segmented_resized)
    side_frame_tk = ImageTk.PhotoImage(side_frame_resized)
    side_segmented_tk = ImageTk.PhotoImage(side_segmented_resized)
    
    label_top_frame.config(image=top_frame_tk)
    label_top_frame.image = top_frame_tk
    label_top_segmented.config(image=top_segmented_tk)
    label_top_segmented.image = top_segmented_tk
    label_side_frame.config(image=side_frame_tk)
    label_side_frame.image = side_frame_tk
    label_side_segmented.config(image=side_segmented_tk)
    label_side_segmented.image = side_segmented_tk
    
    if top_shape:
        if top_shape == "Rectangle":
            top_shape_text = f"Overall Shape: {top_shape}, {top_dimensions[0]:.2f}cm x {top_dimensions[1]:.2f}cm"
        elif top_shape == "Circle":
            top_shape_text = f"Overall Shape: {top_shape}, Diameter: {top_dimensions:.2f}cm"
    else:
        top_shape_text = "No overall shape detected"

    other_shapes_text = "Shapes within:\n"
    for shape, dimensions in top_shapes_within:
        if shape == "Rectangle":
            other_shapes_text += f"{shape}: {dimensions[0]:.2f}cm x {dimensions[1]:.2f}cm\n"
        elif shape == "Circle":
            other_shapes_text += f"{shape}: Diameter {dimensions:.2f}cm\n"

    result_text.set(top_shape_text)
    other_shapes_result_text.set(other_shapes_text)
    lbl_side_result.config(text=f"Object Height: {side_height:.2f} cm")

def capture_all():
    ret_top, top_frame = cap_top.read()
    ret_side, side_frame = cap_side.read()
    if ret_top and ret_side:
        top_processed_frame, top_segmented, top_shape, top_dimensions, top_shapes_within, pixel_to_cm_ratio, x, y, w, h = process_top_frame(top_frame)
        distance_from_box_bottom_cm = calculate_object_distance_from_box_bottom(top_segmented, pixel_to_cm_ratio, y)
        side_height, side_segmented = calculate_object_height(side_frame, AB_cm, distance_from_box_bottom_cm)
        update_gui(top_processed_frame, top_segmented, top_shape, top_dimensions, top_shapes_within, side_frame, side_segmented, side_height)

def show_live_feeds():
    ret_top, top_frame = cap_top.read()
    ret_side, side_frame = cap_side.read()
    if ret_top:
        top_frame_pil = Image.fromarray(cv2.cvtColor(top_frame, cv2.COLOR_BGR2RGB))
        top_frame_resized = top_frame_pil.resize((300, 300))
        top_frame_tk = ImageTk.PhotoImage(top_frame_resized)
        label_top_frame.config(image=top_frame_tk)
        label_top_frame.image = top_frame_tk
    if ret_side:
        side_frame_pil = Image.fromarray(cv2.cvtColor(side_frame, cv2.COLOR_BGR2RGB))
        side_frame_resized = side_frame_pil.resize((300, 300))
        side_frame_tk = ImageTk.PhotoImage(side_frame_resized)
        label_side_frame.config(image=side_frame_tk)
        label_side_frame.image = side_frame_tk

    window.after(10, show_live_feeds)

# Initialize Tkinter window
window = tk.Tk()
window.title("Camera Calibration and Measurement")

# Create layout
frame_top = tk.Frame(window)
frame_top.pack(side="left", padx=10, pady=10)

frame_side = tk.Frame(window)
frame_side.pack(side="right", padx=10, pady=10)

label_top_frame = tk.Label(frame_top)
label_top_frame.pack()
label_top_label = tk.Label(frame_top, text="Top Frame")
label_top_label.pack()

label_top_segmented = tk.Label(frame_top)
label_top_segmented.pack()
label_top_segmented_label = tk.Label(frame_top, text="Top Segmented")
label_top_segmented_label.pack()

label_side_frame = tk.Label(frame_side)
label_side_frame.pack()
label_side_label = tk.Label(frame_side, text="Side Frame")
label_side_label.pack()

label_side_segmented = tk.Label(frame_side)
label_side_segmented.pack()
label_side_segmented_label = tk.Label(frame_side, text="Side Segmented")
label_side_segmented_label.pack()

result_text = tk.StringVar()
other_shapes_result_text = tk.StringVar()

label_result = tk.Label(window, textvariable=result_text, font=("Helvetica", 16, "bold"))
label_result.pack(side="top", pady=10)

label_other_shapes = tk.Label(window, textvariable=other_shapes_result_text, font=("Helvetica", 12))
label_other_shapes.pack(side="top", pady=5)

lbl_side_result = tk.Label(window, text="Object Height: N/A", font=("Helvetica", 14))
lbl_side_result.pack(side="top", pady=5)

button_capture = tk.Button(window, text="Capture", command=capture_all, font=("Helvetica", 14))
button_capture.pack(side="bottom", pady=10)

# Start showing live feeds
show_live_feeds()

# Start Tkinter event loop
window.mainloop()

# Release cameras and close any open windows
cap_top.release()
cap_side.release()
cv2.destroyAllWindows()
