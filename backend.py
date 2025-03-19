"""from flask import Flask, request, jsonify, send_from_directory,  send_from_directory
from flask_cors import CORS"""
import os
from flask import Flask, request, jsonify#, send_from_directory
from flask_cors import CORS
import uuid  # For generating unique file names
from werkzeug.utils import secure_filename  # To handle filenames safely
import cv2  # OpenCV for image processing (object detection)
import numpy as np  # Numpy for numerical operations
import pytesseract  # Tesseract OCR for text recognition (to differentiate between objects and text)
import re  # Regular expressions for text processing

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Server is running!"
"""
# Route to handle Apple touch icons
@app.route('/apple-touch-icon.png')
@app.route('/apple-touch-icon-precomposed.png')
def apple_touch_icon():
    # Serve a default icon or a placeholder image
    return send_from_directory('static', 'default-icon.png')"""
                               
# Use Heroku's dynamic port
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Heroku's assigned port
    app.run(host='0.0.0.0', port=port)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload."""
    print("Upload endpoint hit.")
    
    if "file" not in request.files:
        print("No file part in the request.")
        return jsonify({"message": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        print("Empty filename.")
        return jsonify({"message": "No file selected"}), 400

    if not allowed_file(file.filename):
        print(f"Invalid file type: {file.filename}")
        return jsonify({"message": "Invalid file type"}), 400

    # Secure and generate a unique filename
    original_filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)

    try:
        # Save the file to the server
        file.save(file_path)
        print(f"File saved as: {file_path}")
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        return jsonify({"message": "Failed to save the file."}), 500

    # Process the image
    detected_objects, detected_text = process_image(file_path)
    print("Detected Objects:", detected_objects)
    print("Detected Text:", detected_text)

    return jsonify({
        "message": f"File uploaded successfully as {unique_filename}",
        "file_url": f"/uploads/{unique_filename}",
        "detected_text": detected_text,
        "detected_objects": detected_objects
    }), 200

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(UPLOAD_FOLDER, filename)

def process_image(image_path):
    """Process the uploaded image and return detected text and objects."""
    word_set = set()

    with open("words_alpha.txt","r") as file:
    
        for line in file:
        
            word_set.add(line.strip())
        
    def check_word(word):
        return word in word_set

    #try block to catch errors
    try:

        #loading image
        image = cv2.imread(image_path)

        #image dimensions
        height = 600
        width = 1200

        #checking if the image was loaded properly
        if image is None:
            raise FileNotFoundError("Can not find image file")
    
        #resize image to a smaller size
        image_resize = cv2.resize(image, (width, height))

        #convert image to gray scale
        gray_image = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)

        #load YOLO model
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

        #load class labels
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        #prepare image for object detection (YOLO)
        blob = cv2.dnn.blobFromImage(image_resize, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        #Convert the image into a format suitable for YOLO
        height, width, _ = image_resize.shape
        blob = cv2.dnn.blobFromImage(image_resize, scalefactor=0.00392, size=(416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        #Get YOLO output layers
        layer_names = net.getLayerNames()
        unconnected_layers = net.getUnconnectedOutLayers()
        output_layers = [layer_names[i - 1] for i in unconnected_layers.flatten()]

        #Run YOLO
        outputs = net.forward(output_layers)

        #Filter objects with confidence > 50%
        conf_threshold = 0.5
        detected_objects = ""
        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for detection in output:

                scores = detection[5:]

                class_id = np.argmax(scores)

                confidence = scores[class_id]

                if confidence > conf_threshold:

                    #Storeing all detected objects as string (since we are only interested in one singular object)
                    detected_objects = str(classes[class_id])  

                    center_x, center_y, w, h = (int(detection[0] * width),int(detection[1] * height),int(detection[2] * width),int(detection[3] * height))
                    x = center_x - w // 2
                    y = center_y - h // 2

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                
        #Set Tesseract path (Only needed for Windows)
        #pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        #Apply adaptive thresholding to improve contrast
        gray = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        #Resize to make text larger for OCR
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Use custom Tesseract settings to improve accuracy
        custom_config = r'--oem 3 --psm 6'

        #Run OCR
        detected_text1 = pytesseract.image_to_string(gray, config=custom_config)

        print("Detected Text1:", detected_text1)

        #Enhance image for OCR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        #Run OCR with optimized settings
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        detected_text2 = pytesseract.image_to_string(gray, config=custom_config)
        print("Detected Text2:", detected_text2)

        val1 = detected_text1.split()
        
        val2 = detected_text2.split()

        check1 = []
        
        check2 = []
        
        for i in range(len(val1)):
            val1[i] = re.sub(r"[^a-zA-Z0-9]+", '', val1[i])

        for i in range(len(val1)):

            if (check_word(val1[i].lower()) == True):
                
                check1.append(val1[i])
            
        for i in range(len(val2)):
            val2[i] = re.sub(r"[^a-zA-Z0-9]+", '', val2[i])

        for i in range(len(val2)):

            if (check_word(val2[i].lower()) == True):

                check2.append(val2[i])
        
        if(check2 == []):
            check2.append("")
        
        #Create a set (HashMap) for fast lookup of words and to remove duplicates
        check2_set = set(check2)  # Convert check2 list to a HashSet

        #Adding both sets together to get the final set of words that includes both check1 and check2 without any duplicates or overlapping words
        final_words = set(check1).union(check2_set)

        #Convert set back to string format
        final = "Item name: " + " ".join(final_words)

        return detected_objects, final

    #Catching errors
    except FileNotFoundError as e:
        print("Error:", e)
        return "Error: File not found", []

    except cv2.error as e:
        print("OpenCV Error:", e)
        return "OpenCV Error", []

    except Exception as e:
        print("Unexpected Error:", e)
        return "Unexpected Error", []

if __name__ == "__main__":
    app.run(debug=True, threaded=True)