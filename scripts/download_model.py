import os
import urllib.request

# Create the models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# URL of the YOLOv8n model file
url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
destination = "models/yolov8n.pt"

# Download the model
print("⬇️  Downloading YOLOv8n model...")
urllib.request.urlretrieve(url, destination)
print("✅ YOLOv8n model downloaded and saved to models/yolov8n.pt")
