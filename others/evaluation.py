import torch

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()