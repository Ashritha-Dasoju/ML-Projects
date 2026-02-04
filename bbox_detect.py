from ultralytics import YOLO
import cv2

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Read image
img = cv2.imread("test.jpg")

# Run detection
results = model(img)

# Draw bounding boxes
annotated_frame = results[0].plot()

# Show output
cv2.imshow("Bounding Box Detection", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
