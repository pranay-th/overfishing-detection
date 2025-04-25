import os
from ultralytics import YOLO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def draw_bounding_boxes(image_path, predictions, output_path, num_ships):
    """
    Draw bounding boxes on the image based on YOLO predictions and display the number of ships.
    Args:
        image_path (str): Path to the input image.
        predictions (list): List of predictions from YOLO (bounding boxes, confidence, class).
        output_path (str): Path to save the image with bounding boxes.
        num_ships (int): Number of ships detected in the image.
    """
    # Open the image
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Iterate through predictions and draw bounding boxes
    for pred in predictions:
        box = pred['box']  # Bounding box coordinates
        confidence = pred['confidence']  # Confidence score
        label = pred['class']  # Class label

        # Draw the bounding box
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1] - 10), f"{label} ({confidence:.2f})", fill="red")

    # Add the number of ships to the image
    draw.text((10, 10), f"Fishing Vessels Detected: {num_ships}", fill="yellow")

    # Save the image with bounding boxes
    img.save(output_path)

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted Image with Bounding Boxes (Ships Detected: {num_ships})")
    plt.show()

def predict_ships(model_path, image_dir, output_dir, conf_threshold=0.25):
    """
    Use the YOLOv8 model to predict ships in satellite images and draw bounding boxes.
    Args:
        model_path (str): Path to the YOLOv8 model file (e.g., ship.pt).
        image_dir (str): Directory containing satellite images.
        output_dir (str): Directory to save the prediction results.
        conf_threshold (float): Confidence threshold for predictions.
    """
    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all images in the directory
    for image_name in os.listdir(image_dir):
        if image_name.endswith(('.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            image_path = os.path.join(image_dir, image_name)

            # Perform prediction
            results = model.predict(source=image_path, conf=conf_threshold)

            # Extract predictions
            predictions = []
            for result in results:
                for box in result.boxes:
                    # Extract bounding box coordinates, confidence, and class
                    coords = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
                    confidence = box.conf[0].item()  # Confidence score
                    label = int(box.cls[0].item())  # Class label
                    predictions.append({
                        'box': coords,
                        'confidence': confidence,
                        'class': label
                    })

            # Count the number of ships detected
            num_ships = len(predictions)

            # Draw bounding boxes and save the image
            output_image_path = os.path.join(output_dir, f"predicted_{image_name}")
            draw_bounding_boxes(image_path, predictions, output_image_path, num_ships)

if __name__ == "__main__":
    # Path to the YOLOv8 model
    model_path = "e:/Proj/Overfishing Detection/yolov8_ship_det_satellite/ship.pt"

    # Directory containing satellite images
    image_dir = "E:/Proj/Overfishing Detection/yolov8_ship_det_satellite/images"

    # Directory to save prediction results
    output_dir = "E:/Proj/Overfishing Detection/yolov8_ship_det_satellite/predictions"

    # Run the prediction
    predict_ships(model_path, image_dir, output_dir, conf_threshold=0.25)