**Personalised Advertisemnt**
1. Open command Prompt and go to directory of where the project folder is located 
2. Setup a virtual environment by running 
python -m venv env
 and activate with (.\env\Scripts\activate) for windows and (source env\bin\activate) For MacOs
3. pip install -r requirements.txt <---- this will install all the required library
4. pip install ultralytics to install ultralytics 
5. pip install torch torchvision to install torchvision
6. pip install opencv-python to install opencv
7. run command python app.py 
8. To add/remove advertisments go to static/ads folder and you can add/remove as you may 



## Usage

1. **Start the Flask Application:**
   ```sh
   python3 app.py

2. **Upload a Fast Food Photo:**

- Open your web browser and go to http://localhost:5000.
- Click on the "choose file" button to select image.

3. **View Detected Foods:**

- After selecting the image, click on "Upload and Detect" button and the application will automatically detect the fast food items and display them.  

4. **Adjust Food Quantities:**

- In the results page, you can adjust the quantities for each detected food item based on your preference.

5. **Calculate Total Calories:**

- Click on the "Recalculate Calories" button to get the total calorie content based on the selected foods and their quantities.

6. **View Summary**

- Click on the "View Summary" button to view the summary of the foods calories and the healthy eating tips.

7. **BMI Calculator**

- Click the "BMI Calculator" button to calculate your BMI and to know which class of weight you are.


*Disclaimer*
All rule based for the system to determine the most suitable ads can be edit in the 'app.py' file inside the folder.
Any change to the website can be made from the 'templates' folder located inside the main folder. 
Inside would include all the necassasry files such as about, contact, home, index.
