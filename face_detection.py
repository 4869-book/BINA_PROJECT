import cv2
import os
import shutil
import time
from glob import glob
from PIL import Image
from tqdm import tqdm
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
from torchvision.transforms import ToTensor
import numpy as np

# select device
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"Running on device: {device}")

# Create face detector
mtcnn = MTCNN(
    keep_all=True,
    post_process=True,
    thresholds=[0.6, 0.9, 0.9],
    device=device,
    min_face_size=160,
)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


# หาจำนวนคลัสเตอร์ที่ดีที่สุด
def best_K(x):
    # Define range of k values to test
    k_values = range(2, 11)

    # Calculate silhouette score for each k value
    silhouette_scores = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
        kmeans.fit(x)
        score = silhouette_score(x, kmeans.labels_)
        silhouette_scores.append(score)

    best = index = silhouette_scores.index(max(silhouette_scores)) + 2
    return best


# Resize รูปภาพ
class ResizeUtils:
    # Given a target height, adjust the image by calculating the width and resize
    def rescale_by_height(self, image, target_height, method=cv2.INTER_LANCZOS4):
        """Rescale `image` to `target_height` (preserving aspect ratio)."""
        w = int(round(target_height * image.shape[1] / image.shape[0]))
        return cv2.resize(image, (w, target_height), interpolation=method)

    # Given a target width, adjust the image by calculating the height and resize
    def rescale_by_width(self, image, target_width, method=cv2.INTER_LANCZOS4):
        """Rescale `image` to `target_width` (preserving aspect ratio)."""
        h = int(round(target_width * image.shape[0] / image.shape[1]))
        return cv2.resize(image, (target_width, h), interpolation=method)


# ค้นหาใบหน้า
class FramesGeneratorPickle:
    def __init__(self, FootageSource):
        self.FootageSource = FootageSource

    def AutoResize(self, frame):
        resizeUtils = ResizeUtils()

        height, width, _ = frame.shape

        if height > 500:
            frame = resizeUtils.rescale_by_height(frame, 500)
            self.AutoResize(frame)

        if width > 700:
            frame = resizeUtils.rescale_by_width(frame, 700)
            self.AutoResize(frame)

        return frame

    def GenerateFace(self, OutputDirectoryName, fps):
        v_cap = cv2.VideoCapture(self.FootageSource)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        CurrentDirectory = os.path.curdir
        OutputDirectoryPath = os.path.join(CurrentDirectory, OutputDirectoryName)

        if os.path.exists(OutputDirectoryPath):
            shutil.rmtree(OutputDirectoryPath)
        os.mkdir(OutputDirectoryPath)

        frames = []
        file_count = 0
        for i in tqdm(range(v_len)):
            success = v_cap.grab()
            if i % fps == 0:
                success, frame = v_cap.retrieve()
            else:
                continue
            if not success:
                continue
            # Add to batch

            faces, probs = mtcnn.detect(frame)

            if faces is not None:
                for j, box in enumerate(faces):
                    if probs[j] > 0.8:
                        x1, y1, x2, y2 = box.astype(int)

                        # Get embedding for detected face
                        face = frame[y1:y2, x1:x2]
                        if face.size != 0:
                            face = cv2.resize(face, (160, 160))
                            score = Img_process(face)
                            # print(f'frame {i} face {j} = {score.getScore()}')
                            if score.getScore() > 0:
                                # cv2.imwrite(f"./output/{i}_{j}.jpg", face)
                                img = ToTensor()(face)
                                t = resnet(img.unsqueeze(0).to(device)).squeeze().cpu()
                                # face_embs.append(t)
                                data = {"image": face, "embedded": t}
                                pickle.dump(
                                    data,
                                    open(
                                        f"{OutputDirectoryPath}/image_{file_count}.pkl",
                                        "wb",
                                    ),
                                )
                                cv2.imwrite(
                                    f"./{OutputDirectoryPath}/image{file_count}.jpg",
                                    face,
                                )
                                file_count = file_count + 1
                                # frames.append(data)
        # pickle.dump(frames, open(f"{OutputDirectoryPath}/faceall.pkl", "wb"))

    def clusterFaceAndShow(self, InputPath, ClusterSavePath):
        CurrentDirectory = os.path.curdir
        filepath = os.path.join(CurrentDirectory, InputPath)

        ClusterImgPath = os.path.join(CurrentDirectory, "Clusters")
        if os.path.exists(ClusterImgPath):
            shutil.rmtree(ClusterImgPath)
        os.mkdir(ClusterImgPath)

        # files = pickle.load(open(f"{filepath}/faceall.pkl", "rb"))
        files = sorted(glob("./FramesPickle/*.pkl"))
        faces = []
        images = []
        print(len(files))
        for i, file in enumerate(files):
            data = pickle.load(open(f"{file}", "rb"))
            images.append(data["image"])
            faces.append(data["embedded"].detach().numpy())

        target = int(len(faces) / 2)
        x = PCA(n_components=target).fit_transform(faces)

        model = KMeans(n_clusters=best_K(x), random_state=0, n_init="auto").fit(x)

        labels = model.predict(x)

        examples = []
        for i in range(model.n_clusters):
            cluster_indices = np.where(labels == i)[0]
            distances = np.linalg.norm(
                x[cluster_indices] - model.cluster_centers_[i], axis=1
            )
            closest_index = cluster_indices[np.argmin(distances)]
            cv2.imwrite(f"./{ClusterImgPath}/cluster_{i}.jpg", images[closest_index])
            examples.append(closest_index)

        print(f"[INFO] Got {len(examples)} clusters")
        self.data = x
        self.label = labels

    def scatter(self):
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.label, cmap="viridis")
        plt.show()


class Img_process:
    def __init__(self, ImgInput):
        self.img = ImgInput

    def getScore(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        laplacian_score = min(max(lap_var / 100, 0), 100)
        return laplacian_score


if __name__ == "__main__":
    framesGenerator = FramesGeneratorPickle("../dataset/video7.mp4")
    framesGenerator.GenerateFrame("FramesPickle")  # เปลี่ยนวิดีโอเป็นเฟรม
    framesGenerator.faceDetectSave(
        "FramesPickle", "FacesPickleSave"
    )  # เปลี่ยนวิดีโอเป็นเฟรม
    # framesGenerator.plotShow('FacesPickleSave') # พล็อตโชว์
    framesGenerator.ClusterFace("FacesPickleSave", "Clustered")  # พล็อตโชว์คลัสเตอร์
