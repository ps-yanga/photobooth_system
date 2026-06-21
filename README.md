# 4-Photo Collage Maker
The 4-Photo Collage Maker is a Python-based photobooth application that allows users to capture photos using a webcam or upload images from their device and automatically arrange them into a four-photo collage. Users can customize their collage by applying decorative frames and stickers before saving the final image.

Features
Live camera preview using OpenCV
Capture photos with countdown timer
Upload images from local storage
Automatic 16:9 photo cropping and resizing
Four-photo collage layout
Decorative frame selection
Sticker customization and drag-and-drop functionality
Save final collage as PNG image
Thumbnail preview of captured photos
Undo previously captured photos
Technologies Used
Python 3.x
Tkinter (Graphical User Interface)
OpenCV (Camera Access)
Pillow (Image Processing)
Threading (Camera Operations)
Object-Oriented Programming Concepts Applied
Encapsulation

Data and configuration settings are stored within classes such as Config, FileManager, and PhotoCollage, with methods controlling access and modification.

Inheritance

The CameraThread class inherits from Python's threading.Thread class to implement camera operations in a separate thread.

Polymorphism

Methods such as Image.open() and image processing functions operate on different image types while using the same interface.

Abstraction

Complex operations such as camera initialization, photo processing, and collage generation are hidden behind simple method calls, making the system easier to use.

System Requirements
Windows 10/11
Python 3.10 or higher
Webcam (optional)
At least 4 GB RAM
Required Libraries

Install the required packages using:

pip install pillow opencv-python
Project Structure
project/
│

├── public/

│   ├── assets/

│   │   ├── frames/

│   │   └── stickers/

│   ├── template.png

│   └── logo192.png

│

├── photos/

│   └── saved collages

│

└── main.py

How to Run
Open the project folder.
Install the required libraries.
Run the application:
python main.py
Capture or upload four photos.
Select frames and stickers.
Save the completed collage.
Sample Workflow
Launch the application.
Capture photos using the webcam or upload images.
Add decorative stickers and frames.
Preview the collage.
Save the final image to the photos folder.

Yanga, Princess Sophia A.

BS Computer Engineering
PUP-Manila

This project was developed for educational purposes only.
