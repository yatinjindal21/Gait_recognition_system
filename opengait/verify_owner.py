# import os
# import pickle
# import torch
# import numpy as np
# from opengait.register_owner import extract_embedding_from_pkl, preprocess_sequence, normalize_embedding
# from silhoutte_pre.silhoutte.scripts.sil_vid import process_video


# # -------------------- Verification --------------------
# def temporal_part_similarity(test_emb, owner_emb):
#     """
#     Compute part-level temporal similarity as per GaitPart paper.
#     Args:
#         test_emb: [num_parts, num_frames, feat_dim]
#         owner_emb: [num_parts, num_frames, feat_dim]
#     Returns:
#         final_sim: scalar similarity
#     """
#     num_parts = test_emb.shape[0]
#     part_sims = []

#     for p in range(num_parts):
#         # Ensure 2D: [num_frames, feat_dim]
#         test_part = test_emb[p]
#         owner_part = owner_emb[p]

#         if test_part.dim() == 1:
#             test_part = test_part.unsqueeze(0)
#         if owner_part.dim() == 1:
#             owner_part = owner_part.unsqueeze(0)

#         # Cosine similarity matrix: [T_test, T_owner]
#         sim_matrix = torch.matmul(test_part, owner_part.T).clamp(-1, 1)

#         # For each test frame, take max similarity with owner frames
#         max_per_test_frame = sim_matrix.max(dim=1)[0]
#         part_sims.append(max_per_test_frame.mean().item())

#     # Aggregate across parts
#     final_sim = np.mean(part_sims)
#     return final_sim



# def verify_owner(png_path, cfg_path, model_ckpt, emb_dir='owner_embeddings', threshold=0.9):
#     temp_dir = 'test_cache'
#     pkl_path = preprocess_sequence(png_path, temp_dir)
#     test_emb = extract_embedding_from_pkl(pkl_path, cfg_path, model_ckpt)
#     test_emb = normalize_embedding(test_emb)

#     matched = False
#     for file in os.listdir(emb_dir):
#         if not file.endswith('.pkl'):
#             continue
#         with open(os.path.join(emb_dir, file), "rb") as f:
#             owner_sequences = pickle.load(f)  # list of sequences

#         # Compare against all registered sequences
#         sims = [temporal_part_similarity(test_emb, owner_seq) for owner_seq in owner_sequences]
#         max_sim = max(sims)

#         if max_sim >= threshold:
#             print(f"[SUCCESS] ✅✅✅ Verified as {file} | Similarity: {max_sim:.4f}")
#             matched = True
#         else:
#             print(f"[FAILURE] ❌❌❌ Not matched with {file} | Similarity: {max_sim:.4f}")

#     if not matched:
#         print("[FAILURE] No match found.")
#         return False
#     return True


# if __name__ == "__main__":
#     configPath = "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/opengait/configs/gaitpart/gaitpart.yaml"
#     modelCheckpoint = "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/opengait/output/CASIA-B/GaitPart/GaitPart/checkpoints/GaitPart-120000.pt"
#     test_sequence = "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/datasets/CASIA-B/CASIA-B/006/nm-02/018"
#     verify_owner(test_sequence, configPath, modelCheckpoint)



import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import tempfile
from opengait.register_owner import extract_embedding_from_pkl, preprocess_sequence, normalize_embedding
from silhoutte_pre.silhoutte.scripts.sil_vid import process_video


# -------------------- Temporal Part Similarity --------------------
def temporal_part_similarity(test_emb, owner_emb):
    """
    Compare two gait embeddings (part-based, temporal).
    Both should be tensors of shape [num_parts, feat_dim] or [frames, feat_dim].
    """

    # Convert numpy → torch if needed
    if isinstance(test_emb, np.ndarray):
        test_emb = torch.from_numpy(test_emb)
    if isinstance(owner_emb, np.ndarray):
        owner_emb = torch.from_numpy(owner_emb)

    test_emb = test_emb.float()
    owner_emb = owner_emb.float()

    # Normalize each embedding vector (important for cosine similarity!)
    test_emb = F.normalize(test_emb, p=2, dim=-1)
    owner_emb = F.normalize(owner_emb, p=2, dim=-1)

    # Case 1: embeddings are sequence × feat_dim
    if test_emb.dim() == 2 and owner_emb.dim() == 2:
        # Average across time/parts
        test_emb = test_emb.mean(dim=0, keepdim=True)
        owner_emb = owner_emb.mean(dim=0, keepdim=True)

    # Case 2: embeddings are 1D
    if test_emb.dim() == 1:
        test_emb = test_emb.unsqueeze(0)
    if owner_emb.dim() == 1:
        owner_emb = owner_emb.unsqueeze(0)

    # Cosine similarity
    sim = F.cosine_similarity(test_emb, owner_emb, dim=-1)

    return sim.mean().item()

# -------------------- Verification --------------------
def verify_owner(video_path, cfg_path, model_ckpt, emb_dir='owner_embeddings', threshold=0.9):
    """
    Full pipeline: video → silhouettes → embeddings → verify
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Step 1: Convert video to silhouette frames
        frames_folder = os.path.join(tmp_dir, "frames")
        os.makedirs(frames_folder, exist_ok=True)
        
        username = "test"
        filename = os.path.basename(video_path)
        parts = filename.split("-")
        angle = parts[-1].replace(".avi", "") if len(parts) >= 4 else "0"
        output_base = os.path.join("processed", username, angle)
        frames_folder = os.path.join(output_base, "frames")
        os.makedirs(frames_folder, exist_ok=True)

        processed_video = os.path.join(output_base, filename.replace(".avi", "_silhouette.mp4"))

        process_video(
            video_path,
            output_video=processed_video,       # we don't need processed video file
            output_folder=frames_folder,
            box_size=(128, 128)
        )

        # Step 2: Preprocess silhouettes to PKL (OpenGait format)
        pkl_path = preprocess_sequence(frames_folder, tmp_dir)

        # Step 3: Extract embeddings
        test_emb = extract_embedding_from_pkl(pkl_path, cfg_path, model_ckpt)
        test_emb = normalize_embedding(test_emb)
        
        print("=== DEBUG EMBEDDINGS ===")
        print("test_emb shape:", test_emb.shape)
        print("test_emb first 5:", test_emb.flatten()[:5])


        # Step 4: Compare with all registered owners
        matched = False
        for file in os.listdir(emb_dir):
            if not file.endswith('.pkl'):
                continue

            with open(os.path.join(emb_dir, file), "rb") as f:
                owner_sequences = pickle.load(f)  # list of embeddings per owner sequence

            sims = [temporal_part_similarity(test_emb, owner_seq) for owner_seq in owner_sequences]
            max_sim = max(sims)

            if max_sim >= threshold:
                print(f"[SUCCESS] ✅ Verified as {file.replace('.pkl','')} | Similarity: {max_sim:.4f}")
                matched = True
            else:
                print(f"[FAILURE] ❌ Not matched with {file.replace('.pkl','')} | Similarity: {max_sim:.4f}")

        if not matched:
            print("[FAILURE] No match found.")
            return False
        return True

if __name__ == "__main__":
    input_path = "C:/Users/Dell/OneDrive/Desktop/Capstone/videos_input/VID-20250822-WA0001.mp4"
    configPath = "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/opengait/configs/gaitpart/gaitpart.yaml"
    modelCheckpoint = "C:/Users/Dell/OneDrive/Desktop/Capstone/GaitPart_Implement/opengait/output/CASIA-B/GaitPart/GaitPart/checkpoints/GaitPart-120000.pt"

    verify_owner(input_path, configPath, modelCheckpoint)
