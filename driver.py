from face_detection import *
from face_blur import *
import os

from moviepy.editor import VideoFileClip


def singleton(class_instance):
    instances = {}

    def get_instance(*args, **kwargs):
        if class_instance not in instances:
            instances[class_instance] = class_instance(*args, **kwargs)
        return instances[class_instance]

    return get_instance


@singleton
class Driver:
    def __init__(self) -> None:
        self.input_path = ""
        self.output_path = ""
        self.face_except_list = []
        self.framesGenerator = None
        self.blur = None
        self.mode = 0

    def blur_all_output(self):
        self.blur = Blurring(self.input_path)
        self.blur.BlurAll(self.output_path)
        self.addAudio()

    def blur_except_process(self):
        self.framesGenerator = FramesGeneratorPickle(self.input_path)
        self.framesGenerator.GenerateFace("FramesPickle", 50)
        self.framesGenerator.clusterFaceAndShow("FramesPickle", "Clustered")

    def blur_except_get_face(self):
        return [file for file in os.listdir("./Clusters") if file.endswith(".jpg")]

    def blur_except_output(self):
        self.blur = Blurring(self.input_path)
        self.blur.TargetFace(self.face_except_list)
        self.blur.BlurProcess(self.output_path)
        self.addAudio()

    def addAudio(self):
        video = VideoFileClip(self.input_path)
        audio = video.audio
        out_video = VideoFileClip(self.output_path)
        final_video = out_video.set_audio(audio)

        path = os.path.split(self.output_path)
        final_video.write_videofile(os.path.join(path[0], "blur" + path[1]))
        os.remove(self.output_path)


# driver = Driver()
# print(driver.blur_except_get_face())
# framesGenerator = FramesGeneratorPickle("./videos/video2.mp4")
# framesGenerator.GenerateFace("FramesPickle", 50)  # save face image for cluster
# framesGenerator.clusterFaceAndShow("FramesPickle", "Clustered")  # เก็บคลัสเตอร์

# blur = Blurring(framesGenerator.FootageSource)
# blur.TargetFace([0])  # เลือกหน้า input เป็น ลิสต์
# blur.BlurProcess()  # run blur process output.mp4
