import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Function to classify shape and compute dimensions
def classify_and_measure(contour, pixel_to_cm_ratio):
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    
    if len(approx) == 4:  # Rectangle detection
        x, y, w, h = cv2.boundingRect(approx)
        width_cm = w / pixel_to_cm_ratio
        height_cm = h / pixel_to_cm_ratio
        return "Rectangle", (width_cm, height_cm)
    
    else:  # Circle detection
        area = cv2.contourArea(contour)
        if area > 0:
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area) / (perimeter * perimeter)
            if circularity > 0.75:  # A threshold to confirm it's circular
                (x, y), radius = cv2.minEnclosingCircle(contour)
                diameter_cm = (2 * radius) / pixel_to_cm_ratio
                return "Circle", diameter_cm
    
    return None, None

# Function to process the captured frame
def process_frame(frame):
    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold to isolate the black box from the white background
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

    # Finding contours with hierarchy to capture nested shapes
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Detect the black square by finding the largest contour (based on area)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the black square
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calculate pixel-to-cm ratio based on the size of the black box (assumed to be 10cm x 10cm)
    pixel_size_of_box = max(w, h)  # Use the larger side of the box
    pixel_to_cm_ratio = pixel_size_of_box / 10.0  # Black box is 10cm x 10cm

    # Crop the frame to the region inside the black box
    cropped_frame = frame[y:y+h, x:x+w]
    cropped_thresh = thresh[y:y+h, x:x+w]

    # Detect objects inside the black box using the cropped threshold image
    cropped_contours, cropped_hierarchy = cv2.findContours(cropped_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    shapes_within = []
    largest_shape, largest_dimensions = None, None

    # Process only the contours inside the black box (ignore the outermost contour)
    if cropped_contours:
        # Filter out the largest contour (which corresponds to the black box) and find the next largest contour
        valid_contours = [c for c in cropped_contours if cv2.contourArea(c) < cv2.contourArea(largest_contour)]

        if valid_contours:
            # Find the largest valid shape within the black box
            largest_cropped_contour = max(valid_contours, key=cv2.contourArea)
            largest_shape, largest_dimensions = classify_and_measure(largest_cropped_contour, pixel_to_cm_ratio)

            # Draw the largest contour on the original frame
            cv2.drawContours(cropped_frame, [largest_cropped_contour], -1, (0, 255, 0), 2)

            # Now check for other shapes within the largest shape
            for contour in valid_contours:
                if contour is not largest_cropped_contour:  # Ensure not to double count the largest shape
                    shape, dimensions = classify_and_measure(contour, pixel_to_cm_ratio)
                    if shape and dimensions:
                        shapes_within.append((shape, dimensions))

    return cropped_frame, cropped_thresh, largest_shape, largest_dimensions, shapes_within

# Function to update the Tkinter window with the images and results
def update_gui(frame, edged, largest_shape, largest_dimensions, shapes_within):
    # Convert images to PIL format
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    edged_pil = Image.fromarray(edged)

    # Resize for Tkinter window
    frame_resized = frame_pil.resize((300, 300))
    edged_resized = edged_pil.resize((300, 300))

    # Convert to ImageTk format
    frame_tk = ImageTk.PhotoImage(frame_resized)
    edged_tk = ImageTk.PhotoImage(edged_resized)

    # Update the labels with new images
    label_frame.config(image=frame_tk)
    label_frame.image = frame_tk
    label_edged.config(image=edged_tk)
    label_edged.image = edged_tk

    # Display the largest shape in bigger font and other shapes in smaller font
    if largest_shape:
        if largest_shape == "Rectangle":
            largest_text = f"Overall Shape: {largest_shape}, {largest_dimensions[0]:.2f}cm x {largest_dimensions[1]:.2f}cm"
        elif largest_shape == "Circle":
            largest_text = f"Overall Shape: {largest_shape}, Diameter: {largest_dimensions:.2f}cm"
    else:
        largest_text = "No overall shape detected"

    other_shapes_text = "Shapes within:\n"
    for shape, dimensions in shapes_within:
        if shape == "Rectangle":
            other_shapes_text += f"{shape}: {dimensions[0]:.2f}cm x {dimensions[1]:.2f}cm\n"
        elif shape == "Circle":
            other_shapes_text += f"{shape}: Diameter {dimensions:.2f}cm\n"

    result_text.set(largest_text)
    other_shapes_result_text.set(other_shapes_text)

def capture_frame():
    global frame
    ret, frame = cap.read()
    if ret:
        processed_frame, edged_frame, largest_shape, largest_dimensions, shapes_within = process_frame(frame)
        update_gui(processed_frame, edged_frame, largest_shape, largest_dimensions, shapes_within)

def show_live_feed():
    global frame
    ret, frame = cap.read()
    if ret:
        # Convert the frame to RGB format and resize it for Tkinter
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_resized = frame_pil.resize((300, 300))
        frame_tk = ImageTk.PhotoImage(frame_resized)
        
        # Update the label to show the live feed
        label_frame.config(image=frame_tk)
        label_frame.image = frame_tk

    # Repeat after 10 milliseconds
    window.after(10, show_live_feed)

# Initialize the Tkinter window
window = tk.Tk()
window.title("Shape Detection")

# Set up the Tkinter layout
label_frame = tk.Label(window)
label_frame.pack(side="left", padx=10, pady=10)
label_edged = tk.Label(window)
label_edged.pack(side="left", padx=10, pady=10)

result_text = tk.StringVar()
other_shapes_result_text = tk.StringVar()

label_result = tk.Label(window, textvariable=result_text, font=("Helvetica", 16, "bold"))
label_result.pack(side="top", pady=10)

label_other_shapes = tk.Label(window, textvariable=other_shapes_result_text, font=("Helvetica", 12))
label_other_shapes.pack(side="top", pady=5)

button_capture = tk.Button(window, text="Capture", command=capture_frame, font=("Helvetica", 14))
button_capture.pack(side="bottom", pady=10)

# Open the camera feed
cap = cv2.VideoCapture(0)  # Change to 1 to access the second camera

# Start showing live feed before the button is pressed
show_live_feed()

# Start the Tkinter event loop
window.mainloop()

# Release the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
