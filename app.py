import numpy as np
import cv2
import re
import base64
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from PIL import Image
from io import BytesIO

def getDigits(contours, hierarchy):
    # Get the first index from hierarchy list [[contour 1,contour2,....]]
    # For more details: https://docs.opencv.org/3.4/d9/d8b/tutorial_py_contours_hierarchy.html
    hierarchy = hierarchy[0]
    # Based on retrieval mode used to find contours, for each contour in hierarchy list there is a list having 4 values
    # contour 1/ contour 2 = [Next contour in the hierarchy, previous contour in the hierarchy, Child contour, Parent contour]
    # Create a list of the Parent contour value of all contours and get the list of unique parent contour values
    u, indices = np.unique(hierarchy[:,-1], return_inverse=True)
    # Get common hierarchy value i.e. a parent contour value with most childrens
    most_common_heirarchy = u[np.argmax(np.bincount(indices))]
    # Draw an approximate rectangle around the contours and get list of all such rectangles
    # cv2.boundingRect(ctr) outputs a tuple (x,y,w,h) - origin(x,y), width(w) & height(h) of the bounding rectangle
    bounding_rectangles = [cv2.boundingRect(ctr) for ctr in contours]
    # Define a empty list to store rectangles
    final_bounding_rectangles = []
    # Loop over bounding rectangles & hierarchy
    for r,hr in zip(bounding_rectangles, hierarchy):
        # Get the origin, width & height of the rectangle
        x,y,w,h = r
        # Some rules to filter the rectangles based on area, width, height and the heirarchy level
        if ((w*h)>250) and (10 <= w <= 220) and (10 <= h <= 200) and hr[3] == most_common_heirarchy:
            # Append rectangle to the list
            final_bounding_rectangles.append(r)
    # Return rectangle list
    return final_bounding_rectangles


def processIuput(image):
    # Load the model
    model = load_model('models/model.h5')
    # Convert image to grayscale
    imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Define kernel
    kernel = np.ones((5,5),np.uint8)
    # Apply threshold
    ret,thresh = cv2.threshold(imgray,170,255,0)
    # Apply erosion
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    # Apply dilation
    thresh = cv2.dilate(thresh,kernel,iterations = 1)
    # Apply erosion
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    # Find contours using retrieval mode CV_RETR_TREE & approximation method cv2.CHAIN_APPROX_SIMPLE
    # findContours() Outputs modified image, contours and hierarchy
    _,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Get bounding rectangles for digits in input image 
    digits_rectangles = getDigits(contours,hierarchy)
    # Loop over rectangles
    for rect in digits_rectangles:
        # Get the values from the tuple
        x,y,w,h = rect
        # Draw rectangle on input image
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        # Extract rectangle values from the grayscale version of the image
        im_digit = imgray[y:y+h,x:x+w]
        # Invert image
        im_digit = 255-im_digit
        # Resize the image as (28,28)
        im_digit = np.array(Image.fromarray(im_digit).resize((28, 28), resample=Image.BICUBIC))
        # Process & reshape the image to model input format
        im_digit = (im_digit/255.).reshape(1,28,28,1)
        # Make prediction
        pred = np.argmax(model.predict(im_digit),axis=-1)
        # Write the prediction on input image
        cv2.putText(image, str(int(pred[0])), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
    # Return image
    return image

# Create Flask instance
app = Flask(__name__)

@app.route('/')
def home():
    # Render index page
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request & Convert base64 image data to PIL image
        img = re.sub('^data:image/.+;base64,', '', request.json)
        img = Image.open(BytesIO(base64.b64decode(img)))
        # Convert PIL image to cv2 image
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # Prediction
        out_image = processInput(img)
        # Generate file name
        fname = "static/"+str(np.random.randint(9999,size=1)[0])+".png"
        # Write image
        cv2.imwrite(fname,out_image)
        # Return file name
        return jsonify(result=fname)

if __name__ == "__main__":
    # Run the app
    app.run(debug=True)
