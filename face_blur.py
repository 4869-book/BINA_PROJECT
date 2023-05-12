from face_detection import *

# find distance
def extract_face_frame2(img):
    img = ToTensor()(img)
    embeddings = resnet(img.unsqueeze(0)).detach().cpu()
    return embeddings

def extract_face_target2(img):
    img = ToTensor()(img)
    embeddings = resnet(img.unsqueeze(0)).detach().cpu()
    return embeddings

class Blurring:
    def __init__(self, FootageSource):
        self.FootageSource = FootageSource

    def TargetFace(self, target):
        alltarget = []
        for i in target:
            alltarget += glob(f'./Clustered/{i}/*.pkl')

        embeddeds = []
        for i,file in enumerate(alltarget):
            face = pickle.load(open(file,'rb'))
            embeddeds.append(face['image'])

        self.FaceTarget = embeddeds

    def BlurProcess(self):
        cap = cv2.VideoCapture(self.FootageSource)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define known faces and their names
        known_faces = [
            (extract_face_target2(path)) for path in self.FaceTarget
        ]
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (int(cap.get(3)),int(cap.get(4))))
        
        print(f'[INFO] Start bluring process')
        for _ in tqdm(range(frame_count)):
            ret, frame = cap.read()
            if ret == True:
                # Detect faces in the frame
                boxes, probs = mtcnn.detect(frame)
        
                if boxes is not None:
                    # Iterate over detected faces
                    for i, box in enumerate(boxes):
                        if probs[i] > 0.9:
                            x1, y1, x2, y2 = box.astype(int)
                
                            # Get embedding for detected face
                            face = frame[y1:y2, x1:x2]
                            embedding = extract_face_frame2(cv2.resize(face, (160, 160)))
                
                            # Compare with known faces
                            distances = [np.linalg.norm(embedding - f) for f in known_faces]

                            min_distance = min(distances)
                        
                            if min_distance > 0.5:
                                name = 'unknown'
                                color = (0, 0, 255) # red for unrecognized faces
                                blur = cv2.blur(face, (25, 25))
                                frame[y1:y2, x1:x2] = blur    

                # Save the frame to output video
                out.write(frame)                  
                
            #     cv2.imshow('Frame', frame)
            #     if cv2.waitKey(1) & 0xFF == ord('q'): # press 'q' to exit
            #         break
            # else:
            #     break
        cap.release()
        out.release()
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    blur = Blurring('../dataset/video7.mp4')
    
    blur.TargetFace([0])

    blur.BlurProcess()
    
    
