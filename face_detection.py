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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')

# Create face detector
mtcnn = MTCNN(keep_all=True, post_process=True,thresholds=[0.95, 0.95, 0.95], device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# หาจำนวนคลัสเตอร์ที่ดีที่สุด
def best_K(x):
    # Define range of k values to test
    k_values = range(2, 11)

    # Calculate silhouette score for each k value
    silhouette_scores = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0,n_init="auto")
        kmeans.fit(x)
        score = silhouette_score(x, kmeans.labels_)
        silhouette_scores.append(score)

    best = index = silhouette_scores.index(max(silhouette_scores))+2
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
    
    def GenerateFrame(self, OutputDirectoryName):
        v_cap = cv2.VideoCapture(self.FootageSource)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        CurrentDirectory = os.path.curdir
        OutputDirectoryPath = os.path.join(CurrentDirectory, OutputDirectoryName)

        if os.path.exists(OutputDirectoryPath):
            shutil.rmtree(OutputDirectoryPath)
        os.mkdir(OutputDirectoryPath)

        frames = []
        for i in tqdm(range(v_len)):

            success = v_cap.grab()
            if i % 60 == 0:
                success, frame = v_cap.retrieve()
            else:
                continue
            if not success:
                continue
            # Add to batch

            # frame = self.AutoResize(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
            pickle.dump(frame, open(f'{OutputDirectoryPath}/frame{i}.pkl','wb'))

            # pickle.dump(frames, open(os.path.join(OutputDirectoryPath,'frames.pkl'),'wb'))
        # pickle.dump(frames, open('model.pkl','wb'))

    def faceDetect(self, OutputDirectoryName,SaveDirectoryName): 
        
        CurrentDirectory = os.path.curdir
        OutputDirectoryPath = os.path.join(CurrentDirectory, OutputDirectoryName)
        
        SaveDirectoryPath = os.path.join(CurrentDirectory, SaveDirectoryName)

        if os.path.exists(SaveDirectoryPath):
            shutil.rmtree(SaveDirectoryPath)
        os.mkdir(SaveDirectoryPath)

        paths = glob(f'{OutputDirectoryPath}/*.pkl')
        print('[INFO] Face Detecting')
        face_embs = []
        num = 0
        for i in tqdm(range(len(paths))):
            frame = pickle.load(open(paths[i],'rb'))
            faces = mtcnn(frame)
            if faces is not None:
                for face in faces:
                    t = resnet(face.unsqueeze(0)).squeeze().cpu()
                    # face_embs.append(t)
                    data = {'tensor_embbeding':face ,'embedding':t}
                    pickle.dump(data, open(f'{SaveDirectoryPath}/face{num}.pkl','wb'))
                    num+=1
        print(f'[INFO] Found {num} faces')
        # return face_embs

    def faceDetectSave(self, OutputDirectoryName,SaveDirectoryName): 
        
        CurrentDirectory = os.path.curdir
        OutputDirectoryPath = os.path.join(CurrentDirectory, OutputDirectoryName)
        
        SaveDirectoryPath = os.path.join(CurrentDirectory, SaveDirectoryName)

        if os.path.exists(SaveDirectoryPath):
            shutil.rmtree(SaveDirectoryPath)
        os.mkdir(SaveDirectoryPath)

        paths = glob(f'{OutputDirectoryPath}/*.pkl')
        print('[INFO] Face Detecting')
        face_embs = []
        num = 0
        for i in tqdm(range(len(paths))):
            frame = pickle.load(open(paths[i],'rb'))
            boxes,_ = mtcnn.detect(frame)
            faces = mtcnn(frame)
            if boxes is not None:
                for box, face in zip(boxes,faces):
                    x1, y1, x2, y2 = box.astype(int)
                    image = frame[y1:y2, x1:x2]
                    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    t = resnet(face.unsqueeze(0)).squeeze().cpu().tolist()
                    # cv2.imwrite(f'{SaveDirectoryPath}/face_{num}.jpg', face)
                    data = {'image':cv2.resize(image, (160, 160)) ,'embedded':t}
                    pickle.dump(data, open(f'{SaveDirectoryPath}/face{num}.pkl','wb'))
                    num+=1
        print(f'[INFO] Found {num} faces')
        # return face_embs    

    def plotShow(self,inputpath):
        CurrentDirectory = os.path.curdir
        OutputDirectoryPath = os.path.join(CurrentDirectory, inputpath)

        face_file = glob(f'{inputpath}/*pkl')

        faces = []
        pic = []
        for i,file in enumerate(face_file):
            face = pickle.load(open(file,'rb'))
            faces.append(face['image'])
            pic.append(face['embedded'])
        
        x = PCA(n_components=50).fit_transform(pic)

        plt.scatter(x[:,0], x[:,1], cmap='viridis')
        plt.show()

        # Visualize
        # plt.imshow(faces[0])
        # plt.axis('off')
        # plt.show()
    
    def clusterFaceAndShow(self,inputpath):
        CurrentDirectory = os.path.curdir
        OutputDirectoryPath = os.path.join(CurrentDirectory, inputpath)

        face_file = glob(f'{inputpath}/*pkl')

        faces = []
        pics = []
        for i,file in enumerate(face_file):
            face = pickle.load(open(file,'rb'))
            pics.append(face['image'])
            faces.append(face['embedded'])

        target = int(len(faces)/2)
        x = PCA(n_components=target).fit_transform(faces)

        model = KMeans(n_clusters=best_K(x), random_state=0, n_init="auto").fit(x)

        labels = model.predict(x)

        examples = []
        for i in range(model.n_clusters):
            cluster_indices = np.where(labels == i)[0]
            distances = np.linalg.norm(x[cluster_indices] - model.cluster_centers_[i], axis=1)
            closest_index = cluster_indices[np.argmin(distances)]
            examples.append(closest_index)

        print(f'[INFO] Got {len(examples)} clusters')

        fig, axes = plt.subplots(1, len(examples),figsize=(8, 4))
        for index, ax in zip(examples, axes):
            ax.imshow(pics[index])
        plt.show()

        # plt.scatter(x[:,0], x[:,1], c=labels, cmap='viridis')
        # plt.scatter(centroids[:,0], centroids[:,1], marker="x", color='r')
        # plt.show()

    def ClusterFace(self, InputPath, ClusterSavePath):
        CurrentDirectory = os.path.curdir
        TargetDirectoryPath = os.path.join(CurrentDirectory, InputPath)

        OutputDitrctoryPath = os.path.join(CurrentDirectory, ClusterSavePath)

        if os.path.exists(OutputDitrctoryPath):
            shutil.rmtree(OutputDitrctoryPath)
        os.mkdir(OutputDitrctoryPath)

        face_file = glob(f'{TargetDirectoryPath}/*pkl')

        embeddeds = []
        images = []
        for i,file in enumerate(face_file):
            face = pickle.load(open(file,'rb'))
            images.append(face['image'])
            embeddeds.append(face['embedded'])

        target = int(len(embeddeds)/2)
        x = PCA(n_components=target).fit_transform(embeddeds)

        model = KMeans(n_clusters=best_K(x), random_state=0, n_init="auto").fit(x)

        labels = model.predict(x)

        print(f'[INFO] Saving data')
        for i, label in enumerate(tqdm(labels)):
            data = {'image':images[i] ,'embedded':embeddeds[i], 'label': label}
            
            if not os.path.exists(f'{OutputDitrctoryPath}/{label}'):
                os.mkdir(f'{OutputDitrctoryPath}/{label}')
                
            pickle.dump(data, open(f'{OutputDitrctoryPath}/{label}/face{i}_{label}.pkl','wb'))

        
        

if __name__ == "__main__":

    framesGenerator = FramesGeneratorPickle('../dataset/video7.mp4')
    framesGenerator.GenerateFrame("FramesPickle") #เปลี่ยนวิดีโอเป็นเฟรม
    framesGenerator.faceDetectSave("FramesPickle",'FacesPickleSave') #เปลี่ยนวิดีโอเป็นเฟรม
    # framesGenerator.plotShow('FacesPickleSave') # พล็อตโชว์
    framesGenerator.ClusterFace('FacesPickleSave','Clustered') # พล็อตโชว์คลัสเตอร์