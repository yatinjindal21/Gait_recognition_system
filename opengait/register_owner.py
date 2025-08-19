import os
import argparse
import torch
import pickle
import numpy as np
from pathlib import Path
from ..datasets.pretreatment import imgs2pickle
from main import run_model
from utils import config_loader

parser = argparse.ArgumentParser(description='Main program for opengait.')
parser.add_argument('--cfgs', type=str,
                    default='configs/default.yaml', help="path of config file")
parser.add_argument('--phase', default='train',
                    choices=['train', 'test'], help="choose train or test phase")
parser.add_argument('--log_to_file', action='store_true',
                    help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
parser.add_argument('--iter', default=0, help="iter to restore")
opt = parser.parse_args()

def preprocess_sequence(png_folder, temp_output_dir, sid='owner001'):
    # Hardcoded sequence and view for single-angle uploads
    seq = 'NM01'
    view = '000'
    img_paths = sorted([Path(png_folder)/f for f in os.listdir(png_folder) if f.endswith('.png')])
    imgs2pickle(((sid, 'NM01', '000'), img_paths), Path(temp_output_dir))
    return Path(temp_output_dir)/sid/seq/f"{view}.pkl"


def extract_embedding_from_pkl(pkl_path, cfg_path, model_ckpt):
    cfgs = config_loader(opt.cfgs)
    cfgs.merge_from_file(cfg_path)
    cfgs.freeze()

    model = run_model(cfgs, training=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.eval()

    with open(pkl_path, 'rb') as f:
        seq_data = pickle.load(f)  # shape: [N, H, W]

    seq_data = torch.tensor(seq_data).unsqueeze(0).unsqueeze(0).float()  # [1, 1, N, H, W]
    with torch.no_grad():
        retval = model(seq_data)
    return retval['triplet_feat'].squeeze(0).numpy()  # shape: [dim]


def register_owner(png_path, cfg_path, model_ckpt, save_dir='owner_embeddings'):
    os.makedirs(save_dir, exist_ok=True)
    temp_dir = 'owner_cache'
    pkl_path = preprocess_sequence(png_path, temp_dir)
    embedding = extract_embedding_from_pkl(pkl_path, cfg_path, model_ckpt)

    owner_id = os.path.basename(png_path.rstrip('/'))
    np.save(os.path.join(save_dir, f"{owner_id}.npy"), embedding)
    print(f"[REGISTERED] Owner {owner_id} saved.")
    

def verify_owner(png_path, cfg_path, model_ckpt, emb_dir='owner_embeddings', threshold=0.85):
    temp_dir = 'test_cache'
    pkl_path = preprocess_sequence(png_path, temp_dir, sid='testuser')
    test_emb = extract_embedding_from_pkl(pkl_path, cfg_path, model_ckpt)

    for file in os.listdir(emb_dir):
        if not file.endswith('.npy'):
            continue
        owner_emb = np.load(os.path.join(emb_dir, file))
        sim = np.dot(test_emb, owner_emb) / (np.linalg.norm(test_emb) * np.linalg.norm(owner_emb))
        if sim >= threshold:
            print(f"[SUCCESS] Matched with {file[:-4]} (Similarity: {sim:.4f})")
            return True

    print("[FAILURE] No match found.")
    return False