# mi-kaneyon 1st cretaed
import cv2
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image

# Load the model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model = model.cuda()  # Move the model to GPU
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the background image
background = Image.open('background.jpg')
background = background.resize((640, 480))  # Adjust this to match your webcam resolution
background = np.array(background)  # Keep the background in its original color scale

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to PIL format
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Apply the transformations
    input_tensor = transform(image).unsqueeze(0).cuda()  # Move the input tensor to GPU

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).cpu().numpy()

    # Create a mask and composite image
    mask = output_predictions == 15  # 15 is the label for 'person' in COCO
    mask = np.stack([mask]*3, axis=-1)
    composite = np.where(mask, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), background)  # Keep the frame in its original color scale

    # Convert the composite image back to BGR
    composite = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)

    cv2.imshow('frame', composite)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
