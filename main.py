import os
from PIL import Image
import torch
from gazelle.utils import visualize_heatmap
from gazelle.model import get_gazelle_model

model, transform = get_gazelle_model("gazelle_dinov2_vitl14_inout")
model.load_gazelle_state_dict(torch.load("gazelle_dinov2_vitl14_inout.pt", weights_only=True))
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for inputFilename in os.listdir("/input/"):
    if not inputFilename.split(".")[-1] in ["png","jpg","jpeg"]:
        continue
    image = Image.open("/input/"+inputFilename).convert("RGB")
    input = {
        "images": transform(image).unsqueeze(dim=0).to(device),    # tensor of shape [1, 3, 448, 448]
        "bboxes": [[(0.1, 0.2, 0.5, 0.7)]]              # list of lists of bbox tuples
    }

    with torch.no_grad():
        output = model(input)

    predicted_heatmap = output["heatmap"][0][0]        # access prediction for first person in first image. Tensor of size [64, 64]
    predicted_inout = output["inout"][0][0]            # in/out of frame score (1 = in frame) (output["inout"] will be None  for non-inout models)

    print("Predicted In Out", round(float(predicted_inout), 4))
    viz = visualize_heatmap(image, predicted_heatmap).convert("RGB")
    viz.save("/output/"+inputFilename)