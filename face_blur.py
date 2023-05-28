from face_detection import *


# find distance
def face2emb(img):
    img = cv2.imread(img)
    img = ToTensor()(img)
    embeddings = resnet(img.unsqueeze(0).to(device)).detach().cpu()
    return embeddings


def face2emb2(img):
    img = ToTensor()(img)
    embeddings = resnet(img.unsqueeze(0).to(device)).detach().cpu()
    return embeddings


class Blurring:
    def __init__(self, FootageSource):
        self.FootageSource = FootageSource

    def TargetFace(self, target):
        alltarget = []
        for i in target:
            alltarget += glob(f"./Clusters/cluster_{i}.jpg")

        embeddeds = []
        for i, file in enumerate(alltarget):
            face = face2emb(file)
            embeddeds.append(face)

        self.FaceTarget = embeddeds

    def BlurProcess(self, export_path):
        cap = cv2.VideoCapture(self.FootageSource)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define known faces and their names
        # known_faces = [
        #     (face2emb(path)) for path in self.FaceTarget
        # ]

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(
            export_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4)))
        )

        print(f"[INFO] Start bluring process")
        skip_number = 6
        number_of_frame_skip = skip_number + 1
        is_frame_skip = 0
        face_location = []
        for _ in tqdm(range(frame_count)):
            ret, frame = cap.read()
            is_frame_skip += 1
            if ret == True:
                # Detect faces in the frame
                # if True:
                if is_frame_skip % number_of_frame_skip == 0:
                    print("DETECT frame: ", is_frame_skip)
                    boxes, probs = mtcnn.detect(frame)
                    if boxes is not None:
                        # Iterate over detected faces
                        for i, box in enumerate(boxes):
                            if probs[i] > 0.9:
                                x1, y1, x2, y2 = box.astype(int)

                                # Get embedding for detected face
                                face = frame[y1:y2, x1:x2]
                                if face.size != 0:
                                    compare_face = cv2.resize(face, (160, 160))
                                    embedding = face2emb2(compare_face)

                                    # Compare with known faces
                                    distances = [
                                        np.linalg.norm(embedding - f)
                                        for f in self.FaceTarget
                                    ]
                                    # print(distances)
                                    min_distance = min(distances)

                                    if min_distance > 0.7:
                                        name = "unknown"
                                        color = (
                                            0,
                                            0,
                                            255,
                                        )  # red for unrecognized faces
                                        blur = cv2.blur(face, (25, 25))
                                        frame[y1:y2, x1:x2] = blur
                                        face_location.append([x1, y1, x2, y2])
                else:
                    print("SKIP frame: ", is_frame_skip)
                    for box in face_location:
                        x1 = box[0]
                        y1 = box[1]
                        x2 = box[2]
                        y2 = box[3]
                        blur = cv2.blur(frame[y1:y2, x1:x2], (25, 25))
                        frame[y1:y2, x1:x2] = blur

                    if is_frame_skip % number_of_frame_skip == number_of_frame_skip - 1:
                        face_location = []

                # Save the frame to output video
                out.write(frame)

                # cv2.imshow("Frame", frame)
                # if cv2.waitKey(1) & 0xFF == ord("q"):  # press 'q' to exit
                #     break
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def BlurAll(self, export_path):
        cap = cv2.VideoCapture(self.FootageSource)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define known faces and their names
        # known_faces = [
        #     (face2emb(path)) for path in self.FaceTarget
        # ]

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(
            export_path, fourcc, frame_count, (int(cap.get(3)), int(cap.get(4)))
        )

        print(f"[INFO] Start bluring process")
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

                            blur = cv2.blur(face, (25, 25))
                            frame[y1:y2, x1:x2] = blur

                # Save the frame to output video
                out.write(frame)

                # cv2.imshow("Frame", frame)
                # if cv2.waitKey(1) & 0xFF == ord("q"):  # press 'q' to exit
                #     break
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    blur = Blurring("../dataset/video7.mp4")

    blur.TargetFace([0])

    blur.BlurProcess()
