from face_detection import *
from face_blur import *

framesGenerator = FramesGeneratorPickle('../dataset/video7.mp4')
framesGenerator.GenerateFrame("FramesPickle") #เปลี่ยนวิดีโอเป็นเฟรม
framesGenerator.faceDetectSave("FramesPickle",'FacesPickleSave') #เปลี่ยนวิดีโอเป็นเฟรม
framesGenerator.ClusterFace('FacesPickleSave','Clustered')  #เก็บคลัสเตอร์

blur = Blurring(framesGenerator.FootageSource)   
blur.TargetFace([0]) #เลือกหน้า input เป็น ลิสต์
blur.BlurProcess() #run blur process output.mp4