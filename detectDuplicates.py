import numpy as np
import imagehash
import open_clip
import torch
import cv2
from PIL import Image
import os
import lpips
import torchvision.transforms as T
from tqdm import tqdm
import shutil

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

def compute_lpips_matrix(images, model, device, transform):
    n = len(images)
    sims = np.zeros((n, n), dtype=np.float32)

    tensors = []

    for img in images:

        if isinstance(img, np.ndarray):
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            img = img.convert("RGB")

        t = transform(img).to(device)
        tensors.append(t)

    with torch.no_grad():
        for i in range(n):
            for j in range(i + 1, n):

                d = model(
                    tensors[i].unsqueeze(0),
                    tensors[j].unsqueeze(0)
                )

                sims[i, j] = sims[j, i] = d.item()

    return sims




def detect_duplicates(
    images,
    clip_model,
    preprocess,
    device,
    iou_threshold=1.2, #.95
    phash_threshold=-1, #5
    clip_threshold=1.0, #.95
    activateScore=False,
    scoreThreshold=.7, # Available when activateScore is True
    iou_ratio=0.45, # .5, Available when activateScore is True
    phash_ratio=0.35, # .3, Available when activateScore is True
    clip_ratio=0.2, # .2, Available when activateScore is True
    activateLPSIS=False, # Available when activateScore is True 
    lpsis_model=None, # Available when activateLPSIS is True
    lpips_transform=None, # Available when activateLPSIS is True
    lpsis_ratio=.3, # Available when activateLPSIS is True
    verbose=False
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
    if activateScore and activateLPSIS:
        lpips_matrix = compute_lpips_matrix(images, lpsis_model, device, lpips_transform)

    n = len(images)

    for i in tqdm(range(n), desc="Detecting duplication on images"):
        for j in range(i + 1, n):

            if activateScore:
                iou_s = iou_matrix[i, j]
                phash_s = normalize_phash(phash_matrix[i, j])
                clip_s = clip_matrix[i, j]
                if activateLPSIS:
                    lpips_s = 1.0 - lpips_matrix[i, j]

                score = (
                    iou_ratio * iou_s +
                    phash_ratio * phash_s +
                    clip_ratio * clip_s
                ) if not activateLPSIS else (
                    (iou_ratio - lpsis_ratio/3) * iou_s +
                    (phash_ratio - lpsis_ratio/3) * phash_s +
                    (clip_ratio - lpsis_ratio/3) * clip_s +
                    lpsis_ratio * lpips_s
                )

                if score >= scoreThreshold:
                    if verbose:
                        print(f"[SCORE] {i},{j} = {score:.4f};  (IOU: {iou_s:.4f}, pHash: {phash_s:.4f}, CLIP: {clip_s:.4f}, {f'LPIPS: {lpips_s:.4f}' if activateLPSIS else ''})")
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
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )
    model.to(device)
    model.eval()

    lpips_model = lpips.LPIPS(net='vgg').to(device)

    lpips_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))  # -> [-1,1]
    ])

    dir_path = "Open Comedones_test2"
    images = [Image.open(os.path.join(dir_path, file)) for file in os.listdir(dir_path) if file.endswith((".jpg", ".png", ".jpeg"))]

    clusters = detect_duplicates(
        images,
        model,
        preprocess,
        device,
        iou_threshold=.95, #.95
        phash_threshold=5, #5
        clip_threshold=.97, #.95
        activateScore=True,
        scoreThreshold=.9, # Available when activateScore is True
        iou_ratio=0.4, # .5, Available when activateScore is True
        phash_ratio=0.4, # .3, Available when activateScore is True
        clip_ratio=0.2, # .2, Available when activateScore is True
        activateLPSIS=False, # Available when activateScore is True 
        lpsis_ratio=.3, # Available when activateLPSIS is True
        lpsis_model=lpips_model, # None, Available when activateLPSIS is True
        lpips_transform=lpips_transform, # None, Available when activateLPSIS is True
        verbose=True,
    )

    print(clusters)
    print("Exact images:", len(clusters))

    output_dir = "exact_images"
    lock = False

    save_clusters(lock, clusters, output_dir=output_dir)
    if lock:
        print(f"Exact images saved in '{output_dir}' directory.")