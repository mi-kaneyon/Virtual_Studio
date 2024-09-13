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

cap = cv2.VideoCapture(2)  # Change the camera ID if needed

# Set the video resolution to 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1280, 720))  # Set resolution to 1280x720

recording = False

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
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()  # Convert to numpy array

    # Create a mask and composite image
    mask = output_predictions == 15  # 15 is the label for 'person' in COCO
    mask = np.stack([mask] * 3, axis=-1)
    composite = np.where(mask, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), background)  # Composite with the background

    # Convert the composite image back to BGR for display
    composite = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)

    # Write the frame to the output file if recording
    if recording:
        out.write(composite)

    # Display the output frame
    cv2.imshow('frame', composite)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        recording = not recording  # Toggle recording on/off

cap.release()
out.release()
cv2.destroyAllWindows()
