import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2

# Device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load DeepLabV3 once
print("Loading DeepLabV3 model...")
model = torch.hub.load('pytorch/vision:v0.6.0',
                       'deeplabv3_resnet101',
                       pretrained=True).to(device).eval()

# Person class index in COCO dataset
PEOPLE_CLASS = 15  

# Gaussian blur kernel (for smoothing mask edges)
blur = torch.FloatTensor([[[[1.0, 2.0, 1.0],
                            [2.0, 4.0, 2.0],
                            [1.0, 2.0, 1.0]]]]) / 16.0
blur = blur.to(device)

# Preprocessing normalization (ImageNet stats)
preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

def get_silhouette(image, box_size=(512, 512), threshold_val=127, verbose=False,
                   return_type="silhouette", keep_original_size=True):
    """
    Extract silhouette (white on black) from input image using DeepLabV3.

    Args:
        image (numpy array): Input BGR image.
        box_size (tuple): Resize box size before processing.
        threshold_val (int): Threshold for binarization.
        verbose (bool): Print warnings if no person detected.
        return_type (str): "silhouette" (default) or "mask".
        keep_original_size (bool): Return in original size if True.

    Returns:
        numpy array: Silhouette or mask.
    """
    h, w = image.shape[:2]

    # Resize with padding to fixed box_size
    # fixed_width, fixed_height = box_size
    # scale = min(fixed_width / w, fixed_height / h)
    # new_w, new_h = int(w * scale), int(h * scale)
    # resized = cv2.resize(image, (new_w, new_h))
    # canvas = np.zeros((fixed_height, fixed_width, 3), dtype=np.uint8)
    # x_offset = (fixed_width - new_w) // 2
    # y_offset = (fixed_height - new_h) // 2
    # canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    # image_resized = canvas

    # Convert to tensor
    img_tensor = torch.FloatTensor(image) / 255.0
    input_tensor = preprocess(img_tensor.permute(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
        segmentation = output.argmax(0)

        # Extract only "person" class
        person_mask = segmentation.eq(torch.tensor(PEOPLE_CLASS).to(device)).float()

        if person_mask.sum() == 0:
            if verbose:
                print("⚠️ No person detected in frame")
            return np.zeros((box_size[1], box_size[0], 3), dtype=np.uint8)

        person_mask = person_mask.unsqueeze(0).unsqueeze(0)

        # Smooth mask edges
        for _ in range(3):
            person_mask = F.conv2d(person_mask, blur, stride=1, padding=1)

        # Convert mask to numpy + binarize
        mask = person_mask.squeeze().cpu().numpy()
        _, mask = cv2.threshold((mask * 255).astype(np.uint8),
                                threshold_val, 255, cv2.THRESH_BINARY)

        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, bw, bh = cv2.boundingRect(coords)
            cropped = mask[y:y+bh, x:x+bw]
        # else:
        #     cropped = combined_mask


        resized_mask = cv2.resize(mask, box_size)

        if return_type == "mask":
            return resized_mask
        # Resize back to original size if required
        # if keep_original_size:
        #     combined_mask = cv2.resize(combined_mask, (w, h))

        # if return_type == "mask":
        #     return combined_mask

        # # Generate silhouette (white person on black background)
        silhouette = np.zeros((box_size[1], box_size[0], 3), dtype=np.uint8)
        silhouette[resized_mask == 255] = [255, 255, 255]

        return silhouette
