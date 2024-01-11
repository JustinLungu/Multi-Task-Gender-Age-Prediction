from fastapi import FastAPI, UploadFile, HTTPException, Request, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import torch
from torchvision import transforms
from Gender_Model_Baseline import CustomModel
import io

app = FastAPI()

# Serve static files (e.g., images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Use Jinja2 templates for HTML rendering
templates = Jinja2Templates(directory="templates")


# Load the trained model
model = CustomModel()
model.load_state_dict(torch.load("gender_baseline_weights.pth", map_location=torch.device("cpu")))
model.eval()

# Define the transformation for input images
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


# Define a route to upload an image and get predictions
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # Open the image using PIL
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Transform the image
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)

    # Apply a threshold to get binary prediction
    predicted_class = 1 if output.item() > 0.5 else 0

    # Map gender labels to strings
    gender_mapping = {0: 'Male', 1: 'Female'}
    predicted_gender = gender_mapping.get(predicted_class, 'Unknown')

    return {"predicted_gender": predicted_gender}


# Define a route to show a simple HTML form for image uploading
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("style.html", {"request": request})
