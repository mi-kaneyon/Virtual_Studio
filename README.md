# Virtual_Studio
Personal use simple virtual studio

- Using webcam creating easy broadcasting envrionment.
- Only person tracking and hide your background.
- Don't mind show your realtime video even if background is messy;)


## My environment
Pytorch 2.0.1

torchvision 0.15.2

opencv-python 4.7.0.72

Pillow  9.3.0



## wherever you can take movie! Whenever you want
- Without setting up studio
- just you configure background image file only
- Easy create Virtual studio
- Other use case virtual travel? As you like. 

## command line

```
python studiocam.py

```
Enjoy shooting your DIY movie!

## Replace background image

```

# Load the background image
background = Image.open('background.jpg')
background = background.resize((640, 480))  # Adjust this to match your webcam resolution
background = np.array(background)  # Keep the background in its original color scale

```

## Example
![virtual studio](samplee.png "example")


### Thanks (utilize sample)

https://unsplash.com/photos/VPavA7BBxK0

Photographer Filip Gielda 
