import cv2
import torch
import numpy as np
from torchvision import models
from PIL import Image

# Load the model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model = model.cuda()  # Move the model to GPU
model.eval()

# Define the transformation manually
mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).cuda()
std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).cuda()

# Load the background image
background = Image.open('background.jpg')
background = background.resize((1280, 720))  # Adjust this to match your webcam resolution
background = torch.tensor(np.array(background), dtype=torch.float32).permute(2, 0, 1).cuda()  # Keep the background in its original color scale

cap = cv2.VideoCapture(0)

# Set the video resolution to 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1280, 720))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to tensor format
    image = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).cuda() / 255.0

    # Apply the transformations manually
    input_tensor = (image - mean[:, None, None]) / std[:, None, None]
    input_tensor = input_tensor.unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0)

    # Create a mask and composite image
    mask = output_predictions == 15  # 15 is the label for 'person' in COCO
    mask = mask[None, :, :].repeat(3, 1, 1).float()
    composite = torch.where(mask, image, background)

    # Convert the composite image back to BGR and to CPU
    composite = (composite.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    composite = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)

    # Write the frame to the output file
    out.write(composite)

    cv2.imshow('frame', composite)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
