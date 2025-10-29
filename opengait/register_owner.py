# import os
# import pickle
# import torch
# import torch.nn.functional as F
# import numpy as np
# from pathlib import Path
# from datasets.pretreatment import imgs2pickle
# from opengait.main import build_model
# from opengait.utils import config_loader

# # -------------------- Helpers --------------------
# def preprocess_sequence(png_folder, temp_output_dir):
#     png_folder = Path(png_folder)
#     parts = png_folder.parts
#     if len(parts) >= 3:
#         sid = parts[-3]   # e.g. "001"
#         seq = parts[-2].replace('-', '').upper()  # nm-01 -> NM01
#         view = parts[-1]  # "000"
#     else:
#         raise ValueError(f"Invalid path structure: {png_folder}")

#     img_paths = sorted([png_folder/f for f in os.listdir(png_folder) if f.endswith('.png')])
#     print(f"[DEBUG] Found {len(img_paths)} PNGs for {sid}/{seq}/{view}")

#     temp_output_dir = Path(os.getcwd()) / temp_output_dir
#     temp_output_dir.mkdir(parents=True, exist_ok=True)

#     imgs2pickle(((sid, seq, view), img_paths), temp_output_dir)
#     out_path = temp_output_dir / sid / seq / view / f"{view}.pkl"
#     return out_path.resolve()

# def extract_embedding_from_pkl(pkl_path, cfg_path, model_ckpt):
#     with open(pkl_path, "rb") as f:
#         data = pickle.load(f)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Convert sequence to tensor
#     if isinstance(data, dict):
#         seq = data.get("seq")
#     else:
#         seq = data
#     seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(1).to(device)
#     seq = seq.unsqueeze(0).permute(0, 2, 1, 3, 4)
#     seqL = [seq.shape[2]]

#     # Load model
#     cfgs = config_loader(cfg_path)
#     model = build_model(cfgs, training=False)
#     model._load_ckpt(model_ckpt)
#     model.to(device)
#     model.eval()

#     with torch.no_grad():
#         features = model((seq, None, None, None, seqL))
#         inf_feat = features['inference_feat'] if isinstance(features, dict) else features[0]
#         if isinstance(inf_feat, torch.Tensor):
#             emb = inf_feat[0].cpu()
#         elif isinstance(inf_feat, (list, tuple)):
#             emb = inf_feat[0].cpu()
#         elif isinstance(inf_feat, dict):
#             emb = list(inf_feat.values())[0][0].cpu()
#         else:
#             raise ValueError(f"Unknown inference_feat type: {type(inf_feat)}")

#     return emb  # shape: [num_frames, emb_dim]

# def flatten_and_normalize(emb):
#     """Flatten all frames/sequences and return a single normalized vector [1, emb_dim]."""
#     if isinstance(emb, np.ndarray):
#         emb = torch.tensor(emb, dtype=torch.float32)
#     if emb.dim() > 1:
#         emb = emb.view(-1, emb.shape[-1])  # collapse frames/sequences
#         emb = emb.mean(dim=0, keepdim=True)
#     return F.normalize(emb, dim=1)

# # -------------------- Registration --------------------
# def register_owner_multiple(png_paths, cfg_path, model_ckpt, save_dir='owner_embeddings'):
#     os.makedirs(save_dir, exist_ok=True)
#     temp_dir = 'owner_cache'

#     embeddings = []
#     owner_id = None

#     for png_path in png_paths:
#         pkl_path = preprocess_sequence(png_path, temp_dir)
#         seq_emb = extract_embedding_from_pkl(pkl_path, cfg_path, model_ckpt)
#         seq_emb = flatten_and_normalize(seq_emb)
#         embeddings.append(seq_emb)

#         if owner_id is None:
#             owner_id = os.path.basename(png_path.rstrip('/'))

#     # Average across sequences
#     owner_embedding = torch.cat(embeddings, dim=0).mean(dim=0, keepdim=True)
#     owner_embedding = F.normalize(owner_embedding, dim=1)

#     save_path = os.path.join(save_dir, f"{owner_id}.npy")
#     np.save(save_path, owner_embedding.numpy())
#     print(f"[REGISTERED] Owner {owner_id} with {len(png_paths)} sequences saved at {save_path}.")

# # -------------------- Verification --------------------
# def verify_owner(png_path, cfg_path, model_ckpt, emb_dir='owner_embeddings', threshold=0.9):
#     temp_dir = 'test_cache'
#     pkl_path = preprocess_sequence(png_path, temp_dir)
#     test_emb = extract_embedding_from_pkl(pkl_path, cfg_path, model_ckpt)
#     test_emb = flatten_and_normalize(test_emb)

#     matched = False
#     for file in os.listdir(emb_dir):
#         if not file.endswith('.npy'):
#             continue
#         owner_emb = np.load(os.path.join(emb_dir, file))
#         owner_emb = flatten_and_normalize(owner_emb)

#         sim = F.cosine_similarity(owner_emb, test_emb, dim=1).item()
#         if sim >= threshold:
#             print(f"[SUCCESS] Verified as {file} | Similarity: {sim:.4f}")
#             matched = True
#         else:
#             print(f"[FAILURE] Not matched with {file} | Similarity: {sim:.4f}")

#     if not matched:
#         print("[FAILURE] No match found.")
#         return False
#     return True


# if __name__ == "__main__":
#     configPath = "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/opengait/configs/gaitpart/gaitpart.yaml"
#     modelCheckpoint = "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/opengait/output/CASIA-B/GaitPart/GaitPart/checkpoints/GaitPart-120000.pt"

#     # Register same owner with multiple sequences (videos)
#     owner_sequences = [
#         "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-01/000",
#         "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-02/000",
#         "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-03/000",
#         "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-01/018",
#         "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-02/018",
#         "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-03/018",
#         "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-01/036",
#         "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-02/036",
#         "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-03/036",
#         "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-01/090",
#         "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-02/090",
#         "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-03/090",
#     ]
#     register_owner_multiple(owner_sequences, configPath, modelCheckpoint)

#     # Verify with one test video
#     test_sequence = "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/004/nm-01/000"
#     verify_owner(test_sequence, configPath, modelCheckpoint)

#------------------------------------------------------------------------------------------------------------------------------------------


import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datasets.pretreatment import imgs2pickle
from opengait.main import build_model
from opengait.utils import config_loader

# -------------------- Helpers --------------------
def preprocess_sequence(png_folder, temp_output_dir):
    png_folder = Path(png_folder)
    parts = png_folder.parts
    if len(parts) >= 3:
        sid = parts[-3]   # e.g. "001"
        seq = parts[-2].replace('-', '').upper()  # nm-01 -> NM01
        view = parts[-1]  # "000"
    else:
        raise ValueError(f"Invalid path structure: {png_folder}")

    img_paths = sorted([png_folder/f for f in os.listdir(png_folder) if f.endswith('.png')])
    print(f"[DEBUG] Found {len(img_paths)} PNGs for {sid}/{seq}/{view}")

    temp_output_dir = Path(os.getcwd()) / temp_output_dir
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    imgs2pickle(((sid, seq, view), img_paths), temp_output_dir)
    out_path = temp_output_dir / sid / seq / view / f"{view}.pkl"
    return out_path.resolve()

# def extract_embedding_from_pkl(pkl_path, cfg_path, model_ckpt):
#     with open(pkl_path, "rb") as f:
#         data = pickle.load(f)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Convert sequence to tensor
#     if isinstance(data, dict):
#         seq = data.get("seq")
#     else:
#         seq = data
#     seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(1).to(device)
#     seq = seq.unsqueeze(0).permute(0, 2, 1, 3, 4)
#     seqL = [seq.shape[2]]

#     # Load model
#     cfgs = config_loader(cfg_path)
#     model = build_model(cfgs, training=False)
#     model._load_ckpt(model_ckpt)
#     model.to(device)
#     model.eval()

#     with torch.no_grad():
#         features = model((seq, None, None, None, seqL))
#         inf_feat = features['inference_feat'] if isinstance(features, dict) else features[0]

#         # Shape: [num_parts, num_frames, feat_dim_per_part]
#         if isinstance(inf_feat, torch.Tensor):
#             emb = inf_feat[0].cpu()
#         elif isinstance(inf_feat, (list, tuple)):
#             emb = inf_feat[0].cpu()
#         elif isinstance(inf_feat, dict):
#             emb = list(inf_feat.values())[0][0].cpu()
#         else:
#             raise ValueError(f"Unknown inference_feat type: {type(inf_feat)}")

#     return emb  # Keep part-wise, frame-wise structure


import torch
import torch.nn.functional as F
import numpy as np

def normalize_embedding(emb):
    # Convert to torch tensor if it's numpy
    if isinstance(emb, np.ndarray):
        emb = torch.from_numpy(emb).float().cuda()  # keep on GPU
    elif not torch.is_tensor(emb):
        raise TypeError("Embedding must be a numpy array or torch tensor")

    # L2 normalize
    emb = F.normalize(emb, dim=-1)

    return emb.cpu().numpy()  # back to numpy if needed


# -------------------- Registration --------------------
def register_owner_multiple(png_paths, cfg_path, model_ckpt, save_dir='owner_embeddings'):
    os.makedirs(save_dir, exist_ok=True)
    temp_dir = 'owner_cache'

    embeddings = []
    owner_id = None

    for png_path in png_paths:
        pkl_path = preprocess_sequence(png_path, temp_dir)
        seq_emb = extract_embedding_from_pkl(pkl_path, cfg_path, model_ckpt)
        seq_emb = normalize_embedding(seq_emb)
        embeddings.append(seq_emb)

        if owner_id is None:
            owner_id = os.path.basename(png_path.rstrip('/'))

    # Stack sequences: list of [num_parts, num_frames, feat_dim]
    # We'll store all sequences as-is for proper temporal comparison
    owner_embedding = embeddings

    save_path = os.path.join(save_dir, f"{owner_id}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(owner_embedding, f)

    print(f"[REGISTERED] Owner {owner_id} with {len(png_paths)} sequences saved at {save_path}.")

#==========================================================================================================================================

def preprocess_frames_to_pkl(username, view, frames_folder, temp_output_dir="owner_cache"):
    """Convert frames to pickle (CASIA-B style structure)."""
    frames_folder = Path(frames_folder)
    sid = username
    seq = f"NM01"   # you can keep nm01 as default for registration
    view_dir = str(view).zfill(3)

    img_paths = sorted([frames_folder/f for f in os.listdir(frames_folder) if f.endswith('.png')])
    if not img_paths:
        raise ValueError(f"No PNGs found in {frames_folder}")

    print(f"[DEBUG] Found {len(img_paths)} silhouette frames for {sid}/{seq}/{view_dir}")

    temp_output_dir = Path(temp_output_dir)
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    imgs2pickle(((sid, seq, view_dir), img_paths), temp_output_dir)
    out_path = temp_output_dir / sid / seq / view_dir / f"{view_dir}.pkl"
    return out_path.resolve()



def extract_embedding_from_pkl(pkl_path, cfg_path, model_ckpt):
    # ----- Load sequence from pickle -----
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(data, dict) and "seq" in data:
        seq = data["seq"]  # numpy [T,H,W]
    else:
        seq = data

    seq = torch.tensor(seq, dtype=torch.float32, device=device)  # [T,H,W]
    seq = seq.unsqueeze(1)               # [T,1,H,W]
    seq = seq.unsqueeze(0)               # [1,T,1,H,W]
    seq = seq.permute(0, 2, 1, 3, 4)     # [N=1, C=1, T, H, W]
    seqL = [seq.shape[2]]

    # ----- Build model -----
    cfgs = config_loader(cfg_path)
    model = build_model(cfgs, training=False)
    model._load_ckpt(model_ckpt)
    model.to(device).eval()

    # ----- Forward pass -----
    with torch.no_grad():
        features = model((seq, None, None, None, seqL))

    # ----- Extract inference_feat -----
    if isinstance(features, dict) and "inference_feat" in features:
        inf_feat = features["inference_feat"]
    else:
        inf_feat = features

    # Helper to find tensor
    def first_tensor(x):
        if torch.is_tensor(x):
            return x
        if isinstance(x, (list, tuple)):
            for item in x:
                if torch.is_tensor(item):
                    return item
        if isinstance(x, dict):
            for k in ["embeddings", "local", "feat", "features"]:
                if k in x and torch.is_tensor(x[k]):
                    return x[k]
            for k in sorted(x.keys()):
                if torch.is_tensor(x[k]):
                    return x[k]
        raise ValueError(f"Could not find a tensor in inference_feat; got type {type(x)}")

    emb = first_tensor(inf_feat).detach().cpu()  # shape can vary

    # ----- Standardize shape -----
    if emb.dim() == 4:
        # [N, P, T, D] → take first batch
        emb = emb[0]   # [P, T, D]
    elif emb.dim() == 3:
        if emb.shape[0] == 1:
            emb = emb[0]   # [T, D] → treat as [1, T, D] later
        else:
            # Already [P, T, D]
            pass
    elif emb.dim() == 2:
        emb = emb.unsqueeze(0)  # [1, T, D]
    else:
        raise ValueError(f"Unexpected embedding shape: {emb.shape}")

    # ----- Pool across time dimension -----
    # Average across frames to get a single embedding vector per part
    if emb.dim() == 3:  # [P, T, D]
        emb = emb.mean(dim=1)   # [P, D]

    # If there are multiple parts, average them
    if emb.dim() == 2:  # [P, D]
        emb = emb.mean(dim=0)   # [D]

    # ----- Normalize embedding -----
    emb = F.normalize(emb, p=2, dim=0)   # unit vector

    return emb.numpy()  # [D]



def register_owner_from_frames(username, view, frames_folder, cfg_path, model_ckpt, save_dir="owner_embeddings"):
    os.makedirs(save_dir, exist_ok=True)

    # Convert silhouette frames → pickle
    pkl_path = preprocess_frames_to_pkl(username, view, frames_folder)

    # Extract embedding
    emb = extract_embedding_from_pkl(pkl_path, cfg_path, model_ckpt)

    # Save embedding
    save_path = os.path.join(save_dir, f"{username}.pkl")
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            existing_embs = pickle.load(f)
    else:
        existing_embs = []

    existing_embs.append(emb)
    with open(save_path, "wb") as f:
        pickle.dump(existing_embs, f)

    print(f"[REGISTERED] {username} with {len(existing_embs)} sequences. Saved at {save_path}.")
    return save_path



# -------------------- Main --------------------
if __name__ == "__main__":
    configPath = "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/opengait/configs/gaitpart/gaitpart.yaml"
    modelCheckpoint = "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/opengait/output/CASIA-B/GaitPart/GaitPart/checkpoints/GaitPart-120000.pt"

    # Register same owner with multiple sequences
    owner_sequences = [
        "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-01/000",
        "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-02/000",
        "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-03/000",
        "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-01/018",
        "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-02/018",
        "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-03/018",
        "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-01/036",
        "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-02/036",
        "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-03/036",
        "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-01/054",
        "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-02/054",
        "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-03/054",
        "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-01/072",
        "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-02/072",
        "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-03/072",
        "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-01/090",
        "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-02/090",
        "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/001/nm-03/090",
    ]
    register_owner_multiple(owner_sequences, configPath, modelCheckpoint)

    # Verify with one test video
    test_sequence = "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/006/nm-02/018"
