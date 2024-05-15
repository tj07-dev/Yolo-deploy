import os
import platform
import base64
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket 
from pydantic import BaseModel
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Any
from yoloDetection.pipeline.training_pipeline import TrainPipeline
from yoloDetection.utils.main_utils import decodeImage, encodeImageIntoBase64
from yoloDetection.constant.application import APP_HOST, APP_PORT
from collections import Counter
import yolov9Mod
import json
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

class BaseResponse(BaseModel):
    status: str
    body: Optional[Any]
    message: Optional[str]

# Load the model outside the function scope
model = yolov9Mod.load('./yolov9Mod/bestmodel.pt', device="cpu")

# Read the CSV file once and store it in memory
csv_file_path = "./data/Food_Nutrition.csv"
df = pd.read_csv(csv_file_path)

app = FastAPI()

model.conf = 0.25  # NMS confidence threshold
model.iou = 0.6  # NMS IoU threshold
current_os = platform.system()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClientApp:
    def __init__(self):
        self.filename = "input.jpg"

clApp = ClientApp()

@app.post("/train")
def train_route():
    obj = TrainPipeline()
    obj.run_pipeline()
    return "Training Successfull!!"

@app.get("/", response_class=HTMLResponse)
def home():
    content = """
    <html>
        <body>
            <h1>Welcome to the Home Page</h1>
        </body>
    </html>
    """
    return HTMLResponse(content=content)

@app.post("/predict", response_model=BaseResponse)
async def predict_route(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Process the image data
        image_data = await process_image_data(contents)

        # Set the model's confidence and IoU thresholds
        model.conf = 0.25
        model.iou = 0.6

        # Perform the prediction
        results = model(image_data)
        json_result = json.loads(results.pandas_to_json())

        # Check for the specific error message
        if "error" in json_result and json_result["error"] == "No data available in xyxy[0]":
            return BaseResponse(status="no_detection", message="No product found", body=None)

        fruit_names = [fruit['name'] for fruit in json_result]
        fruit_counts = Counter(fruit_names)
        fruit_counts_dict = dict(fruit_counts)

        # Filter and process the CSV data
        filtered_df = df[df["Fruit"].isin(fruit_counts_dict.keys())].copy()
        filtered_df["Quantity"] = filtered_df["Fruit"].apply(lambda fruit: fruit_counts_dict[fruit])
        for nutrient in filtered_df.columns:
            if nutrient not in ["Fruit", "Quantity"]:
                filtered_df[nutrient] = filtered_df[nutrient] * filtered_df["Quantity"]

        all_fruit_info = json.loads(filtered_df.to_json(orient="records"))

        # Save the prediction result image
        results.save()

        # Encode the prediction result image to base64
        opencodedbase64 = encodeImageIntoBase64("./runs/detect/exp/input.jpg")
        result = {"detection": all_fruit_info, "image": opencodedbase64.decode('utf-8')}

        # Delete the runs directory
        delete_runs_directory()

        return BaseResponse(status="success", message="Product found", body=result)
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return BaseResponse(status="error", message="An error occurred during prediction",  body=None)


@app.websocket("/wspredict")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            # Process the image data
            image_data = await process_image_data(data)

            # Perform the prediction
            results = model(image_data)
            json_result = json.loads(results.to_json())

            if "error" in json_result and json_result["error"] == "No data available in xyxy[0]":
                await websocket.send_json({"status":"no_detection", "message":"No product found", "body":None}) 

            # Send the prediction result back to the WebSocket client
            else :
                await websocket.send_json({"status":"success", "message":"Product found", "body":json_result})

            # Delete the runs directory
            delete_runs_directory()

    except Exception as e:
        await websocket.send_json({"status": "error", "message": f"WebSocket Error: {e}", "body": None})
        logging.error(f"WebSocket Error: {e}")
        await websocket.close()


def delete_runs_directory():
    # Delete the runs directory
    runs_path = "./runs"
    if os.path.exists(runs_path):
        shutil.rmtree(runs_path)

async def process_image_data(contents):
    # Check if the contents are in base64 format
    if contents.startswith(b'data:image'):
        # Extract the base64 data
        base64_data = contents.split(b',')[1]
        # Decode the base64 data
        image_data = base64.b64decode(base64_data)
    else:
        # If the contents are not base64, treat them as binary data
        image_data = contents

    # Save the image data to a file
    with open("./data/" + clApp.filename, 'wb') as f:
        f.write(image_data)

    return "./data/" + clApp.filename

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)