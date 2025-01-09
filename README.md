# Fast Food Calorie Estimation Project

This project is designed to use the YOLO (You Only Look Once) model to identify fast food items, estimate their calorie content, and provide a user interface through a Flask web application.
![Ekran görüntüsü 2024-08-03 145456](https://github.com/user-attachments/assets/35bd2ab0-209d-463f-a7d2-80535a3ba532)


## About the Project

In this project, we have developed a system that uses artificial intelligence technology to estimate the calorie content of fast food. The system allows users to upload food photos, adjust the quantities of identified foods, and calculate total calories.

### Features

- Image-based detection of fast food items
- Calorie content estimation of detected foods
- User-friendly web interface
- Multiple food selection and quantity adjustment
- Total calorie calculation

## Dataset and Model Training


### Data Collection
We created a comprehensive fast food image dataset for our project. This dataset includes images collected from various fast food restaurants and online sources.


### Data Labeling
The collected images were labeled according to the YOLO format. This process involved determining the locations and classes of each food item in every image.
I use makesense.ai in labeling.
![Ekran görüntüsü 2024-08-03 162745](https://github.com/user-attachments/assets/73fe01ec-d16c-4bae-94c0-13e9e929fd1f)


### Model Training
![Ekran görüntüsü 2024-08-03 162848](https://github.com/user-attachments/assets/8620a8d4-4e9c-4292-b8e4-f906c45dc8ea)

The YOLO model was trained on the labeled dataset. Our model can detect the following fast food items:
- Pizza
- Burger
- French Fries
- Nuggets
- Cola
- Hot Dog
- Onion Rings

## Application Features

![Ekran görüntüsü 2024-08-03 141314](https://github.com/user-attachments/assets/94da7bf3-307b-435a-8634-0b5c3e6f8b2a)


Our application offers users the following features:
1. Image upload: Users can upload a fast food photo they want to analyze.
2. Automatic food detection: Food items in the uploaded image are automatically detected.
3. Quantity adjustment: Users can adjust the quantity for each detected food item.
4. Calorie calculation: Total calorie content is calculated based on the selected foods and quantities.
5. Results display: Detected foods, quantities, and total calorie information are shown to the user.

## Project Structure

```
.
├── images/                 # All training and testing images
├── labels/                 # Labeled image annotations
├── results/                # Test results
├── resultsOfApp/           # Results generated by the Flask application
├── runs/                   # Trained model files
│   └── detect/
│       └── train/
│           └── weights/
│               └── best.pt # Best trained model
├── templates/              # Flask application templates
│   └── index.html          # Main page template
├── uploadsOfApp/           # Images uploaded to the Flask application
├── yolo_format/            # Data format created for YOLO
├── app.py                  # Flask application file
├── fastfood.yaml           # YOLO configuration file
├── labelList.txt           # List of labels
├── main.py                 # Main script (data preparation, training, testing ,caloriecalculation)
└── yolov8n.pt              # YOLOv8 pre-trained model
```



## Installation on Ubuntu

- sudo apt update
- sudo apt upgrade
- sudo apt install python3
- sudo apt install build-essential
- sudo apt install cmake ninja-build
- sudo apt install python3-pip
- pip install torch torchvision
- pip install opencv-python
- pip install ultralytics
- git clone https://github.com/busenurileri/CalorieDetectionProject.git
- cd CalorieEstimationProject
If you want, you can create your own dataset in YOLO format using the appropriate functions in `main.py`, or you can directly run `app.py` to start the application.In order to run a python file write in command line:
- python3 "filename.py"

## Usage

1. **Start the Flask Application:**
   ```sh
   python3 app.py

2. **Upload a Fast Food Photo:**

- Open your web browser and go to http://localhost:5000.
- Click on the "Dosya seç" button to select image.

3. **View Detected Foods:**

- After selecting the image, click on "Upload and Detect" button and the application will automatically detect the fast food items and display them.  

4. **Adjust Food Quantities:**

- In the results page, you can adjust the quantities for each detected food item based on your preference.

5. **Calculate Total Calories:**

- Click on the "Recalculate Calories" button to get the total calorie content based on the selected foods and their quantities.
