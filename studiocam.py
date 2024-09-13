# mi-kaneyon 2nd cretaed
import cv2
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image

# Load the model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check if GPU is available
model = model.to(device)  # Move the model to GPU if available
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the background image
background = Image.open('background.jpg')
background = background.resize((1280, 720))  # Adjust this to match your webcam resolution
background = np.array(background)  # Keep the background in its original color scale

cap = cv2.VideoCapture(2)

# Set the video resolution to 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to PIL format
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Apply the transformations
    input_tensor = transform(image).unsqueeze(0).to(device)  # Move the input tensor to GPU if available

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)['out'][0]  # Perform inference on the model
    output_predictions = output.argmax(0).byte().cpu().numpy()  # Move to CPU and convert to numpy array

    # Create a mask and composite image
    mask = output_predictions == 15  # 15 is the label for 'person' in COCO
    mask = np.stack([mask] * 3, axis=-1)  # Convert mask to 3-channel format
    composite = np.where(mask, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), background)  # Composite with the background

    # Convert the composite image back to BGR for display
    composite = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)

    # Show the output frame
    cv2.imshow('frame', composite)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
