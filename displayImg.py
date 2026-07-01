import cv2
import os

dir_path = "Open Comedones_test2"

def display_images(lst, idxs, window_name="Images"):
    idxs = list(idxs)
    i = 0

    while True:
        idx = idxs[i]
        img = cv2.imread(os.path.join(dir_path, lst[idx]))

        if img is None:
            print(f"Image not found: {lst[idx]}")
            i += 1
            if i >= len(idxs):
                break
            continue

        cv2.imshow(window_name, img)

        key = cv2.waitKeyEx(0)

        # ESC
        if key == 27:
            break

        # LEFT arrow
        elif key in [2424832, 65361]:
            i = max(0, i - 1)

        # RIGHT arrow
        elif key in [2555904, 65363]:
            i += 1
            if i >= len(idxs):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    lst = os.listdir(dir_path)
    prompt = input("Enter image indices (comma-separated): ")
    indices = [int(i) for i in prompt.split(",")]

    display_images(lst, indices)
    for idx in indices:
        print(f"Image {idx}: {lst[idx]}")