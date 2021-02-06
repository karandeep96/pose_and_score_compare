import argparse
from calculations import get_Score
import cv2 as cv
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument("-a", "--activity", required=True,
	help="activity to be scored")
ap.add_argument("-v1", "--video1", required=True,
	help="video file to be scored against")
ap.add_argument("-v2", "--video2", required=True,
	help="video file to be scored")
ap.add_argument("-l", "--lookup", default="lookup_test.pickle",
	help="The pickle file containing the lookup table")
ap.add_argument('--thr', default=0.2, type=float, 
	help='Threshold value for pose parts heat map')
ap.add_argument('--width', default=368, type=int, 
	help='Resize input to specific width.')
ap.add_argument('--height', default=368, type=int, 
	help='Resize input to specific height.')
	
args = vars(ap.parse_args())

g = get_Score(args["lookup"])

final_score,score_list = g.calculate_Score(args["video2"],args["activity"])
print(final_score)
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

inWidth = args["width"]
inHeight = args["height"]

net = cv.dnn.readNetFromTensorflow("model-mobilenet_v1_101.pb")

cap1 =cv.VideoCapture(args["video1"] if args["video1"] else 0)
'''while cv.waitKey(1) < 0:
    hasFrame1, frame1 = cap1.read()
    if not hasFrame1:
        cv.waitKey()
        break
    cv.imshow('Pose Output', frame1)'''
    
    
cap2 = cv.VideoCapture(args["video2"] if args["video2"] else 0)


while cv.waitKey(1) < 0:
    hasFrame, frame = cap2.read()
    if not hasFrame:
        cv.waitKey()
        break
    hasFrame1, frame1 = cap1.read()
    if not hasFrame1:
        cv.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > args["thr"] else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    #cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv.putText(frame, 'Score : ' + str(final_score), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2)
    cv.imshow('Original', frame1)
    cv.imshow('Pose Output', frame)
    
