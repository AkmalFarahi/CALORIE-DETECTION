from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename
import os
from pathlib import Path
from ultralytics import YOLO
import cv2
#import requests  # Add requests library for Nutritionix API
import pandas as pd

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploadsOfApp'
RESULT_FOLDER = 'resultsOfApp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = "runs/detect/train/weights/best.pt"  # Update the model path if needed

# Nutritionix API credentials
#API_KEY = "55bad3032798745ae268a9dd2080cffe"
#APP_ID = "1aa56b2c"


# Initialize YOLO model
try:
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading YOLO model:", e)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Create folders if they don't exist
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(RESULT_FOLDER).mkdir(exist_ok=True)

# Calorie information for detected items (Fallback values)
calorie_dict = {
    'pizza': 285,
    'burger': 250,
    'friedpatato': 365,
    'nugget': 300,
    'cola': 150,
    'hotdog': 150,
    'onionring': 200
}

# Nutritional data for known items
nutrition_data = {
    'pizza': {
        "calories": 285,
        "protein": 12,
        "fat": 10,
        "carbs": 36
    },
    'burger': {
        "calories": 250,
        "protein": 12,
        "fat": 9,
        "carbs": 30
    },
    'friedpatato': {
        "calories": 365,
        "protein": 3,
        "fat": 17,
        "carbs": 50
    },
    'nugget': {
        "calories": 300,
        "protein": 14,
        "fat": 15,
        "carbs": 30
    },
    'cola': {
        "calories": 150,
        "protein": 0,
        "fat": 0,
        "carbs": 39
    },
    'hotdog': {
        "calories": 150,
        "protein": 5,
        "fat": 12,
        "carbs": 3
    },
    'onionring': {
        "calories": 200,
        "protein": 2,
        "fat": 12,
        "carbs": 20
    }
}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_nutritional_info(food_item):
    """
    Fetch nutritional information for a given food item.
    Currently a placeholder returning local data.
    """
    return nutrition_data.get(food_item.lower(), {"error": "Nutritional info not available"})

    
def predict_image(image_path):
    """Run YOLOv8 model on an image and calculate total calories."""
    try:
        img = cv2.imread(image_path)
        results = model(img)
    except Exception as e:
        print(f"Error during image prediction: {e}")
        return [], 0, None

    detected_items = []
    total_calories = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            c = box.cls
            class_name = model.names[int(c)]
            calories = calorie_dict.get(class_name, 0)
            
            # Calculate total calories (this could be updated based on actual quantities or data from the nutrition API)
            detected_items.append({'name': class_name, 'calories': calories})
            total_calories += calories

            # Draw bounding boxes and labels on the image
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{class_name}: {calories} kcal'
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save the result image
    result_path = os.path.join(app.config['RESULT_FOLDER'], 'result_' + os.path.basename(image_path))
    cv2.imwrite(result_path, img)
    
    return detected_items, total_calories, result_path

# Additional Healthy Eating Features
nutrition_advice = {
    'pizza': "Consider reducing portion size or opting for a thin crust and more vegetables.",
    'burger': "Choose lean meats and whole-grain buns to make it healthier.",
    'friedpatato': "Try baked or air-fried potatoes instead of deep-frying.",
    'nugget': "Consider grilling chicken or using plant-based protein alternatives.",
    'cola': "Replace with water or unsweetened beverages to reduce sugar intake.",
    'hotdog': "Opt for whole-grain buns and add vegetables like spinach or tomatoes.",
    'onionring': "Bake instead of deep-frying, or use a healthier batter with less oil."
    }

def calculate_bmi(weight, height):
    try:
        # Convert height from cm to meters by dividing by 100
        height_in_meters = height / 100
        
        # Calculate BMI: weight (kg) / (height_in_meters^2)
        return weight / (height_in_meters ** 2)
    except ZeroDivisionError:
        return None  # Return None in case of zero height

    
def classify_bmi(bmi_value):
    """Classify the BMI and provide weight advice."""
    if bmi_value is None:
        return None, None

    # Classify BMI
    if bmi_value < 18.5:
        bmi_classification = "Underweight"
        weight_classification = "You should increase calorie intake."
    elif 18.5 <= bmi_value < 24.9:
        bmi_classification = "Normal weight"
        weight_classification = "Maintain your current weight."
    elif 25 <= bmi_value < 29.9:
        bmi_classification = "Overweight"
        weight_classification = "Consider a balanced diet and exercise."
    else:
        bmi_classification = "Obese"
        weight_classification = "Consult a healthcare provider for weight management."

    return bmi_classification, weight_classification


@app.route('/index', methods=['GET'])
def index():
    """Render the index.html template."""
    return render_template('index.html')

@app.route('/', methods=['GET'])
def main_page():
    """Redirect to index for the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle image upload and run YOLO model."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            detected_items, total_calories, result_path = predict_image(filepath)
            
            # Fetch nutritional breakdown from Nutritionix API
            nutrition_info = {}
            for item in detected_items:
                food_item = item['name']
                print(f"Fetching nutritional info for: {food_item}")  # Debugging line
                nutrition = get_nutritional_info(food_item)
                if nutrition and 'error' not in nutrition:
                    nutrition_info[food_item] = nutrition
                else:
                    nutrition_info[food_item] = {"error": "Nutritional info not available"}

            return jsonify({
                'detected_items': detected_items,
                'total_calories': total_calories,
                'nutrition_info': nutrition_info,
                'result_image': os.path.basename(result_path)
            })
    return render_template('upload.html')

@app.route('/results/<filename>')
def uploaded_file(filename):
    """Serve the processed result image."""
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/nutrition/<item_name>')
def nutrition(item_name):
    """Fetch and display the nutritional information for a given food item."""
    item_info = nutrition_data.get(item_name.lower())  # case insensitive
    if item_info:
        return render_template('nutrition.html', item=item_name.capitalize(), info=item_info)
    else:
        return "Nutritional information not available for this item", 404
    
@app.route('/recalculate', methods=['POST'])
def recalculate_calories():
    """Recalculate total calories based on updated weights."""
    data = request.json
    total_calories = 0
    for item in data:
        calories_per_100g = calorie_dict.get(item['name'], 0)
        total_calories += (calories_per_100g * item['grams']) / 100
    return jsonify({'total_calories': round(total_calories, 2)})

@app.route('/live')
def live_feed():
    """Render the live feed template."""
    return render_template('live.html')

def generate_live_feed():
    """Generate live video feed with YOLO detection."""
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam

    if not cap.isOpened():
        print("Error: Webcam not accessible!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame not captured.")
            break

        # Perform detection
        results = model(frame)
        for r in results:
            for box in r.boxes:
                c = box.cls
                class_name = model.names[int(c)]
                calories = calorie_dict.get(class_name, 0)

                # Draw bounding boxes and labels
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{class_name}: {calories} kcal'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode the frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    """Provide the video feed as a streaming response."""
    return Response(generate_live_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/healthy_eating_advice', methods=['POST'])
def healthy_eating_advice():
    """
    Endpoint to provide healthy eating advice for detected items.
    Expects JSON data with a list of detected items.
    """
    data = request.json  # Should include detected items like [{'name': 'pizza'}, {'name': 'cola'}, ...]
    advice = []

    for item in data:
        name = item.get('name')
        grams = item.get('grams', None)  # Optional grams information
        advice_text = nutrition_advice.get(name, "Enjoy in moderation as part of a balanced diet.")

        item_advice = {
            'name': name,
            'advice': advice_text
        }
        if grams:
            item_advice['grams'] = grams

        advice.append(item_advice)

    return jsonify({'healthy_eating_advice': advice})

@app.route('/summary', methods=['POST'])
def summary():
    """
    Combines detected items, total calories, and healthy eating advice into a single summary.
    Expects JSON data with detected items and their grams.
    """
    data = request.json  # Expected format: [{'name': 'pizza', 'grams': 150}, ...]
    total_calories = 0
    advice = []

    for item in data:
        name = item['name']
        grams = item.get('grams', 100)  # Default to 100 grams if not provided
        calories_per_100g = calorie_dict.get(name, 0)
        item_calories = (calories_per_100g * grams) / 100
        total_calories += item_calories

        advice_text = nutrition_advice.get(name, "Enjoy in moderation as part of a balanced diet.")
        advice.append({
            'name': name,
            'grams': grams,
            'calories': round(item_calories, 2),
            'advice': advice_text
        })

    return jsonify({
        'total_calories': round(total_calories, 2),
        'items_with_advice': advice
    })

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')

@app.route('/bmi_calc', methods=['POST', 'GET'])
def bmi_calc():
    bmi = None
    bmi_classification = None
    weight_classification = None
    bmi_value = None  # Add this to store the raw numeric BMI value

    if request.method == 'POST':
        try:
            # Convert inputs to float
            weight = float(request.form['weight'])
            height = float(request.form['height'])

            if weight <= 0 or height <= 0:
                return render_template('bmi_calc.html', error="Weight and height must be positive values.")

            # Calculate BMI
            bmi_value = calculate_bmi(weight, height)

            # Format the BMI to 2 decimal places
            if bmi_value is not None:
                bmi = "Your BMI is: %.2f" % bmi_value
            else:
                bmi = "Invalid BMI"

            # Classify BMI and weight
            bmi_classification, weight_classification = classify_bmi(bmi_value)

        except (ValueError, KeyError) as e:
            print("Error:", e)
            return render_template('bmi_calc.html', error="Please enter valid numeric values for weight and height")

    # Pass both the formatted BMI (bmi) and raw BMI value (bmi_numeric_value) to the template
    return render_template('bmi_calc.html', bmi=bmi, bmi_numeric_value=bmi_value, bmi_classification=bmi_classification, weight_classification=weight_classification)



if __name__ == '__main__':
    app.run(debug=True)
