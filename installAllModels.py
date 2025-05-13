import torch
from gazelle.utils import visualize_heatmap
from gazelle.model import get_gazelle_model

model, transform = get_gazelle_model("gazelle_dinov2_vitb14")
model.load_gazelle_state_dict(torch.load("gazelle_dinov2_vitb14.pt", weights_only=True))
model.eval()

model, transform = get_gazelle_model("gazelle_dinov2_vitl14")
model.load_gazelle_state_dict(torch.load("gazelle_dinov2_vitl14.pt", weights_only=True))
model.eval()

model, transform = get_gazelle_model("gazelle_dinov2_vitb14_inout")
model.load_gazelle_state_dict(torch.load("gazelle_dinov2_vitb14_inout.pt", weights_only=True))
model.eval()

model, transform = get_gazelle_model("gazelle_dinov2_vitl14_inout")
model.load_gazelle_state_dict(torch.load("gazelle_dinov2_vitl14_inout.pt", weights_only=True))
model.eval()
