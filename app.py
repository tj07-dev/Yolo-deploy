import sys
import os
import platform

from yoloDetection.pipeline.training_pipeline import TrainPipeline
from yoloDetection.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template,Response
from flask_cors import CORS, cross_origin
from yoloDetection.constant.application import APP_HOST, APP_PORT
# from yolov9.detect import run
from ultralytics import YOLO
from collections import Counter
import yolov9Mod
import json
import pandas as pd


model = yolov9Mod.load(
    './yolov9/bestmodel.pt',
    device="cpu"
)

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.6  # NMS IoU threshold
current_os = platform.system()

# model = YOLO('./yolov9/models/detect/gelan-c.yaml')
# model = YOLO('./yolov9/bestmodel.pt') 
csv_file_path = "./data/Food_Nutrition.csv"


app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"



@app.route("/train")
def trainRoute():
    obj = TrainPipeline()
    obj.run_pipeline()
    return "Training Successfull!!" 


@app.route("/")
def home():
    return render_template("index.html")



@app.route("/predict", methods=['POST','GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        print(type(image))
        decodeImage(image, clApp.filename)

        # os.system("cd yolov9/ && python detect.py --weights bestmodel.pt --conf 0.3 --save-conf --save-txt --source ../data/inputImage.jpg")
        results = model(r'./data/inputImage.jpg')
        json_result = json.loads(results.pandas_to_json())
        # Extract the names of the fruits from the results
        fruit_names = [fruit['name'] for fruit in json_result]
        # Count each unique fruit type
        fruit_counts = Counter(fruit_names)
        # Convert the Counter object to a regular dictionary
        fruit_counts_dict = dict(fruit_counts)
        # Output the dictionary with the counts
        print(fruit_counts_dict)
        
        df = pd.read_csv(csv_file_path)

        filtered_df = df[df["Fruit"].isin(fruit_counts_dict.keys())].copy()

        filtered_df["Quantity"] = filtered_df["Fruit"].apply(lambda fruit: fruit_counts_dict[fruit])

        # Multiply nutrition values by quantity
        for nutrient in filtered_df.columns:
            if nutrient not in ["Fruit", "Quantity"]:  # Exclude non-nutrition columns
                filtered_df[nutrient] = filtered_df[nutrient] * filtered_df["Quantity"]

        # Create a single JSON response containing all filtered fruit information
        all_fruit_info = json.loads(filtered_df.to_json(orient="records"))

        # Print the combined JSON response
        print(all_fruit_info)

        image = results.save()
        opencodedbase64 = encodeImageIntoBase64("./runs/detect/exp/inputImage.jpg")
        # result = {"image": image, "detection":all_fruit_info}
        result = {"image": opencodedbase64.decode('utf-8'), "detection":all_fruit_info}
        # os.system("rm -rf ./runs")
        # os.system("rm -rf yolov9/runs")
        if current_os == "Windows":
            os.system("rmdir /S /Q runs")
        elif current_os == "Linux" or current_os == "Darwin":  # Linux or macOS
            os.system("rm -rf runs")
        else:
            print("Unsupported operating system.")

    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"

    return jsonify(result)



@app.route("/live", methods=['GET'])
@cross_origin()
def predictLive():
    try:
        os.system("cd yolov9/ && python detect.py --weights my_model.pt --img 416 --conf 0.2 --source 0")
        # os.system("rm -rf yolov9/runs")
        if current_os == "Windows":
            os.system("rmdir /S /Q runs")
        elif current_os == "Linux" or current_os == "Darwin":  # Linux or macOS
            os.system("rm -rf runs")
        else:
            print("Unsupported operating system.")
        # os.system("del -rf yolov9/runs")
        return "Camera starting!!" 

    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")

@app.route("/export", methods=['GET'])
def export():
    try:
        os.system("cd yolov9/ && python export.py --weights bestmodel.pt --include tflite")
        # os.system("rm -rf yolov9/runs")
        if current_os == "Windows":
            os.system("rmdir /S /Q runs")
        elif current_os == "Linux" or current_os == "Darwin":  # Linux or macOS
            os.system("rm -rf runs")
        else:
            print("Unsupported operating system.")
        # os.system("del -rf yolov9/runs")
        # return "Camera starting!!" 

    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    
if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host=APP_HOST, port=APP_PORT)

