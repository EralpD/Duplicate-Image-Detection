from PIL import Image
import imagehash
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import open_clip

add1 = "./images/identical/copy1.jpg"
add2 = "./images/identical/copy2.jpg"

diff1 = "./images/different/image1.png"
diff2 = "./images/different/image2.png"

img1 = Image.open(add1)
img2 = Image.open(add2)

diffImg1 = Image.open(diff1)
diffImg2 = Image.open(diff2)

def display_image(img):
    img.show()

def check_Identical(img1, img2):
    hash1 = imagehash.phash(img1)
    hash2 = imagehash.phash(img2)
    distance = hash1 - hash2
    if distance == 0:
        print("The images are identical.")
    else:
        print("The images are not identical.")

def random_crop(img, crop_ratio=0.6):
    h, w = img.shape[:2]

    nh, nw = int(h * crop_ratio), int(w * crop_ratio)

    y = np.random.randint(0, h - nh)
    x = np.random.randint(0, w - nw)

    cropped = img[y:y+nh, x:x+nw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)

def freq_distort(img):
    blur = cv2.GaussianBlur(img, (9,9), 2)
    sharpen = cv2.addWeighted(img, 1.8, blur, -0.8, 0)
    return sharpen

def resample_jitter(img):
    h, w = img.shape[:2]

    img = cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_LINEAR)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(img, (int(w*0.8), int(h*0.8)), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    return img

def color_shift(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv[:,:,0] = (hsv[:,:,0] + np.random.randint(0, 50)) % 180
    hsv[:,:,1] = np.clip(hsv[:,:,1] * np.random.uniform(0.5, 1.5), 0, 255)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def warp(img):
    h, w = img.shape[:2]

    pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    pts2 = np.float32([
        [0,0],
        [w*0.9, h*0.1],
        [w*0.1, h*0.9],
        [w, h]
    ])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h))

def jpeg_cycle(img):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), np.random.randint(30, 90)]
    _, enc = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(enc, 1)

def _to_numpy(img):

    if not isinstance(img, np.ndarray):
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img

def compute_iou(img1, img2):

    img1 = _to_numpy(img1)
    img2 = _to_numpy(img2)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))

    _, m1 = cv2.threshold(img1, 0, 255, cv2.THRESH_OTSU)
    _, m2 = cv2.threshold(img2, 0, 255, cv2.THRESH_OTSU)

    m1 = m1.astype(bool)
    m2 = m2.astype(bool)

    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()

    if union == 0:
        return 0.0

    return inter / union

def compute_clip_similarity(img1, img2):

    img1 = _to_numpy(img1)
    img2 = _to_numpy(img2)

    img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    img1 = preprocess(img1).unsqueeze(0).to(device)
    img2 = preprocess(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        feat1 = model.encode_image(img1)
        feat2 = model.encode_image(img2)

        feat1 = F.normalize(feat1, dim=-1)
        feat2 = F.normalize(feat2, dim=-1)

        similarity = (feat1 * feat2).sum(dim=-1).item()

    return similarity

def preprocess_size(img, scale=0.5):
    h, w = img.shape[:2]

    new_w = int(w * scale)
    new_h = int(h * scale)

    img = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    return img

def preprocess_resolution(img, max_dim=4096): # Downscales the resolution of the image
    h, w = img.shape[:2]

    scale = min(max_dim / w, max_dim / h)

    if scale >= 1:
        return img

    new_w = int(w * scale)
    new_h = int(h * scale)

    img = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    return img

def check_Similar(img1, img2, func, iou_threshold=0.85, p_threshold=5, clip_threshold=.95, save=False, verbose=True):      
    """
    Priority-based duplicate detection:
    1. IOU
    2. pHash 
    3. CLIP 
    """

    img2 = cv2.imread(add2)
    if func:
        img2 = func(img2)
    img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    if verbose:
        display_image(img1)
        display_image(img2)

    if save:
        if func:
            img2.save(f"./images/preprocessedImgs/copy_{func.__name__}.jpg")
        else:
            pass

    iou_score = compute_iou(img1, img2)
    if iou_score is not None and iou_score > iou_threshold:
        print(f"The images are similar. (IOU score: {iou_score:.3f})")
        return

    phash1 = imagehash.phash(img1)
    phash2 = imagehash.phash(img2)
    distance = phash1 - phash2

    if distance <= p_threshold:
        print("The images are similar. (phash distance: {})".format(distance))
        return

    cliP_score = compute_clip_similarity(img1, img2)
    if cliP_score is not None and cliP_score > clip_threshold:
        print(f"The images are similar. (CLIP score: {cliP_score:.3f})")
        return

    print("The images are not similar. (IOU score: {}, phash distance: {}, CLIP score: {})".format(iou_score, distance, cliP_score))


# check_Identical(hash1, hash2)
# check_Similar(img1, img2, preprocess_size, threshold=5) # The distance = 0
# check_Similar(img1, img2, preprocess_resolution, threshold=5) # The distance = 0
stack = []
stack.append((check_Similar, img1, img2, random_crop, .85, 5, .95, True, True)) # The distance = 34 (broken); idx 0
stack.append((check_Similar, img1, img2, freq_distort, .85, 5, .95, True, True)) # The distance = 0; idx 1
stack.append((check_Similar, img1, img2, resample_jitter, .85, 5, .95, True, True)) # The distance = 0; idx 2
stack.append((check_Similar, img1, img2, color_shift, .85, 5, .95, True, True)) # The distance = 6 (kindaly not similar); idx 3
stack.append((check_Similar, img1, img2, warp, .85, 5, .95, True, True)) # The distance = 18 (broken); idx 4
stack.append((check_Similar, img1, img2, jpeg_cycle, .85, 5, .95, True, True)) # The distance = 0; idx 5

wanted = None
# wanted = {0, 3, 4} # The broken ones (random_crop, color_shift, warp) are detected similar by IoU: (0.865, 0.999, 0.955)

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai"
    )

    model = model.to(device)
    model.eval()

    if wanted:
        stack = [stack[i] for i in wanted]
        for func, *args in stack:
            print(f"Running {func.__name__}...")
            func(*args)

    else:
        func, *args = stack.pop()
        print(f"Running {func.__name__}...")
        func(*args)

    print("\nChecking on Different Images:\n")

    check_Similar(diffImg1, diffImg2, None, .85, 5, .95, True, True)
