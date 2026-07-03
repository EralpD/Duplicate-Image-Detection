import numpy as np
import imagehash
import open_clip
import torch
import cv2
from PIL import Image
import os
# import lpips
import torchvision.transforms as T
from tqdm import tqdm
import shutil
import argparse


try:
    from pytorch_msssim import ssim
    PT_SSIM_AVAILABLE = True
except ImportError:
    from skimage.metrics import structural_similarity as compare_ssim 
    print("pytorch_msssim is not installed, or GPU not been available. Please install it using 'pip install pytorch-msssim'.")
    PT_SSIM_AVAILABLE = False

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)

        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

def build_clip_embeddings(images, model, preprocess, device, batch_size=32):
    processed = []

    for img in images:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        tensor = preprocess(img)
        processed.append(tensor)

    processed = torch.stack(processed)

    embs = []

    with torch.no_grad():
        for i in range(0, len(processed), batch_size):
            batch = processed[i:i+batch_size].to(device)

            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)

            embs.append(features.cpu())

    embs = torch.cat(embs, dim=0)
    return embs.numpy()

def compute_clip_similarity_matrix(embeddings):
    return embeddings @ embeddings.T

def build_phashes(images):
    hashes = []

    for img in images:
        if isinstance(img, np.ndarray):
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            pil = img

        hashes.append(imagehash.phash(pil))

    return hashes

def compute_phash_matrix(phashes):
    n = len(phashes)
    mat = np.zeros((n, n), dtype=np.int32)

    for i in range(n):
        for j in range(i, n):
            d = phashes[i] - phashes[j]
            mat[i, j] = d
            mat[j, i] = d

    return mat

def normalize_phash(dist):
    return max(0.0, 1.0 - (dist / 64.0))  # 64-bit pHash

def build_iou_masks(images):
    masks = []
    for img in images:
        arr = _to_numpy(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256))
        _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_OTSU)
        masks.append(mask.astype(np.bool_))

    return np.stack(masks, axis=0)

def _to_numpy(img):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def to_tensor(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    img = preprocess(img)
    return img.unsqueeze(0).to(device)

def compute_iou_matrix(masks):
    n = masks.shape[0]

    flat = masks.reshape(n, -1).astype(np.float32)

    intersections = flat @ flat.T
    areas = flat.sum(axis=1)

    unions = areas[:, None] + areas[None, :] - intersections
    iou = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections),
        where=unions != 0
    )

    return iou

# def compute_lpips_matrix(images, model, device, transform):
#     n = len(images)
#     sims = np.zeros((n, n), dtype=np.float32)

#     tensors = []

#     for img in images:

#         if isinstance(img, np.ndarray):
#             if img.shape[-1] == 4:
#                 img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

#             img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         else:
#             img = img.convert("RGB")

#         t = transform(img).to(device)
#         tensors.append(t)

#     with torch.no_grad():
#         for i in range(n):
#             for j in range(i + 1, n):

#                 d = model(
#                     tensors[i].unsqueeze(0),
#                     tensors[j].unsqueeze(0)
#                 )

#                 sims[i, j] = sims[j, i] = d.item()

#     return sims

def compute_ssim_matrix_gpu(images, device):
    n = len(images)
    tensors = []
    for img in images:
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        # 1 kanallı gri tonlamaya çevirip normalize ediyoruz
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        gray = cv2.resize(gray, (256, 256))
        tensors.append(torch.tensor(gray, dtype=torch.float32) / 255.0)
    
    # Batch formatı: [N, 1, 256, 256]
    batch = torch.stack(tensors).unsqueeze(1).to(device)
    
    if PT_SSIM_AVAILABLE:
        # GPU üzerinde vektörize SSIM matrisi (Deneysel & Çok Hızlı)
        ssim_matrix = torch.eye(n, device=device)
        with torch.no_grad():
            for i in tqdm(range(n), desc="Computing SSIM matrix"):
                # i. görseli tüm görsellerle tek seferde karşılaştır
                img_i = batch[i:i+1].expand(n, -1, -1, -1)
                # ssim fonksiyonu batch bazlı çalışabilir (size_average=False ile her çift için skor döner)
                scores = ssim(img_i, batch, data_range=1.0, size_average=False)
                ssim_matrix[i] = scores
        return ssim_matrix
    else:
        # pytorch-msssim yoksa eski CPU yöntemine mecburen geri döner
        sims = np.zeros((n, n), dtype=np.float32)
        resized = [t.cpu().numpy() for t in tensors]
        for i in tqdm(range(n), desc="Computing SSIM matrix"):
            sims[i, i] = 1.0
            for j in range(i + 1, n):
                s = compare_ssim(resized[i], resized[j], data_range=1.0)
                sims[i, j] = sims[j, i] = s
        return torch.tensor(sims, device=device)
    
def build_orb_descriptors(images):
    orb = cv2.ORB_create(nfeatures=256)
    descriptors = []

    for img in images:
        arr = _to_numpy(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        _, des = orb.detectAndCompute(gray, None)
        descriptors.append(des)

    return descriptors

def compute_orb_similarity_matrix(descriptors):
    n = len(descriptors)
    sims = np.zeros((n, n), dtype=np.float32)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for i in tqdm(range(n), desc="Computing ORB matrix"):
        sims[i, i] = 1.0

        for j in range(i + 1, n):
            des1 = descriptors[i]
            des2 = descriptors[j]

            if des1 is None or des2 is None:
                sim = 0
            else:
                matches = bf.match(des1, des2)

                if len(matches) == 0:
                    sim = 0
                else:
                    good = [m for m in matches if m.distance < 32]
                    sim = len(good) / max(len(des1), len(des2))

            sims[i, j] = sims[j, i] = sim

    return sims




def detect_duplicates(
    images,
    clip_model,
    preprocess,
    device,
    iou_threshold=0.95, 
    phash_threshold=5,
    clip_threshold=0.97,
    activateScore=True,
    scoreThreshold=1, # Available when activateScore is True
    iou_ratio=0.4, # .5, Available when activateScore is True
    phash_ratio=0.4, # .3, Available when activateScore is True
    clip_ratio=0.2, # .2, Available when activateScore is True
    activateSSIM=True, # Available when activateScore is True
    ssim_ratio=0.3, # Available when activateSSIM is True
    activateORB=True, # Available when activateScore is True
    orb_ratio=0.2, # Available when activateORB is True
    verbose=True 
):
    n = len(images)
    uf = UnionFind(n)
    masks = build_iou_masks(images)
    phashes = build_phashes(images)
    embeddings = build_clip_embeddings(
        images,
        clip_model,
        preprocess,
        device
    )

    iou_matrix = compute_iou_matrix(masks)
    clip_matrix = compute_clip_similarity_matrix(embeddings)
    phash_matrix = compute_phash_matrix(phashes)
    if activateScore and activateSSIM:
        ssim_matrix = compute_ssim_matrix_gpu(images, device)
    if activateScore and activateORB:
        orb_descriptors = build_orb_descriptors(images)
        orb_matrix = compute_orb_similarity_matrix(orb_descriptors)

    n = len(images)

    for i in tqdm(range(n), desc="Detecting duplication on images"):
        for j in range(i + 1, n):

            if activateScore:
                iou_s = iou_matrix[i, j]
                phash_s = normalize_phash(phash_matrix[i, j])
                clip_s = clip_matrix[i, j]
                if activateSSIM:
                    ssim_s = ssim_matrix[i, j]
                if activateScore and activateORB:
                    orb_s = orb_matrix[i, j] if activateORB else 0

                score = (
                    (iou_ratio-ssim_ratio/3 if activateSSIM else iou_ratio) * iou_s +
                    (phash_ratio-ssim_ratio/3 if activateSSIM else phash_ratio) * phash_s +
                    (clip_ratio-ssim_ratio/3 if activateSSIM else clip_ratio) * clip_s +
                    (ssim_ratio * ssim_s if activateSSIM else 0)
                )

                if activateORB:
                    score = score * (1 - orb_ratio) + orb_s * orb_ratio

                if score >= scoreThreshold:
                    if verbose:
                        print(f"[SCORE] {i},{j} = {score:.4f};  (IOU: {iou_s:.4f}, pHash: {phash_s:.4f}, CLIP: {clip_s:.4f}, {f'SSIM: {ssim_s:.4f}' if activateSSIM else ''}, {f'ORB: {orb_s:.4f}' if activateORB else ''})")
                    uf.union(i, j)

            else:
                is_dup = False

                if iou_matrix[i, j] > iou_threshold:
                    if verbose:
                        print(f"[IOU] {i},{j} = {iou_matrix[i, j]:.4f}")
                    is_dup = True

                elif phash_matrix[i, j] <= phash_threshold:
                    if verbose:
                        print(f"[pHash] {i},{j} = {phash_matrix[i, j]:.4f}")
                    is_dup = True

                elif clip_matrix[i, j] > clip_threshold:
                    if verbose:
                        print(f"[CLIP] {i},{j} = {clip_matrix[i, j]:.4f}")
                    is_dup = True

                if is_dup:
                    uf.union(i, j)

    clusters = {}

    for i in range(n):
        root = uf.find(i)

        if root not in clusters:
            clusters[root] = []

        clusters[root].append(i)

    return clusters

def save_clusters(lock, clusters, output_dir="exact_images"):
    if lock:
        os.makedirs(output_dir, exist_ok=True)

        images = os.listdir(dir_path)

        for i,cluster_idx in enumerate(clusters.keys()):
            try:
                file_name = images[cluster_idx]

                _, ext = os.path.splitext(file_name)

                new_file_name = f"images_{i}{ext}"

                src_path = os.path.join(dir_path, file_name)
                dst_path = os.path.join(output_dir, new_file_name)

                shutil.copy(src_path, dst_path)

            except Exception as e:
                print(f"Error copying file {file_name}: {e}")
    else:
        pass

        


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()

    parser.add_argument("--dir_path", type=str)

    parser.add_argument("--iou_threshold", type=float, default=0.95)
    parser.add_argument("--phash_threshold", type=int, default=5)
    parser.add_argument("--clip_threshold", type=float, default=0.97)

    parser.add_argument("--activateScore", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--scoreThreshold", type=float, default=.9)

    parser.add_argument("--iou_ratio", type=float, default=0.4)
    parser.add_argument("--phash_ratio", type=float, default=0.4)
    parser.add_argument("--clip_ratio", type=float, default=0.2)

    parser.add_argument("--activateSSIM", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--ssim_ratio", type=float, default=0.3)

    parser.add_argument("--activateORB", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--orb_ratio", type=float, default=0.2)

    parser.add_argument("--verbose", type=lambda x: x.lower() == "true", default=True)

    args = parser.parse_args()

    if not args.dir_path:
        raise ValueError("Please provide a valid directory path using --dir_path argument.")
    


    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )
    model.to(device)
    model.eval()

    # lpips_model = lpips.LPIPS(net='vgg').to(device)

    # lpips_transform = T.Compose([
    # T.Resize((256, 256)),
    # T.ToTensor(),
    # T.Normalize((0.5, 0.5, 0.5),
    #             (0.5, 0.5, 0.5))  # -> [-1,1]
    # ])

    dir_path = args.dir_path
    images = [Image.open(os.path.join(dir_path, file)) for file in os.listdir(dir_path) if file.endswith((".jpg", ".png", ".jpeg"))]

    clusters = detect_duplicates(
        images,
        model,
        preprocess,
        device,
        iou_threshold=args.iou_threshold,
        phash_threshold=args.phash_threshold,
        clip_threshold=args.clip_threshold,
        activateScore=args.activateScore,
        scoreThreshold=args.scoreThreshold,
        iou_ratio=args.iou_ratio,
        phash_ratio=args.phash_ratio,
        clip_ratio=args.clip_ratio,
        activateSSIM=args.activateSSIM,
        ssim_ratio=args.ssim_ratio,
        activateORB=args.activateORB,
        orb_ratio=args.orb_ratio,
        verbose=args.verbose,
    )

    print(clusters)
    print("Exact images:", len(clusters))

    output_dir = "exact_images"
    lock = False

    save_clusters(lock, clusters, output_dir=output_dir)
    if lock:
        print(f"Exact images saved in '{output_dir}' directory.")
