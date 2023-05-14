from face_detection import *
from face_blur import *

framesGenerator = FramesGeneratorPickle('../dataset/video7.mp4')
# framesGenerator.GenerateFace("FramesPickle",50) #save face image for cluster
framesGenerator.clusterFaceAndShow('FramesPickle','Clustered')  #เก็บคลัสเตอร์

# blur = Blurring(framesGenerator.FootageSource)   
# blur.TargetFace([0]) #เลือกหน้า input เป็น ลิสต์
# blur.BlurProcess() #run blur process output.mp4