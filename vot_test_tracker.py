from tracker.re3_tracker import Re3Tracker
import numpy as np
import os
import utils.IOU as IOU
import utils.drawing as drawing
import cv2

SEQ_PATH = 'vot2014-back/sequences/'

def to_xyxy(input):
    x_coord = input[[0,2,4,6]]
    y_coord = input[[1,3,5,7]]
    bbox = [np.min(x_coord), np.min(y_coord), np.max(x_coord), np.max(y_coord)]
    return np.array(bbox)

if __name__ == "__main__":

    VIDEO_PATH = [SEQ_PATH + f + '/' for f in os.listdir(SEQ_PATH)]
    VIDEO_PATH.sort()

    scores = []

    for video in VIDEO_PATH:
        
        tracker = Re3Tracker()
        print('Now in folder: ' + video)
        image_path = [video + 'color/' + f for f in os.listdir(video + 'color/')]
        image_path.sort()
        ground_truth = np.loadtxt(video + 'groundtruth.txt', delimiter=',')
        
        img = cv2.imread(image_path[0])
        bbox = to_xyxy(ground_truth[0,:])
        bbox = tracker.track(1, img, bbox)
        
        for i in range(1, len(image_path)):
            img = cv2.imread(image_path[i])
            predicted_bbox = tracker.track(1,img)
            score = IOU.IOU(predicted_bbox, to_xyxy(ground_truth[i,:]))
            print('score:', score)
            scores.append(score)
    
    print('Save scores to file ...')
    
    with open('IOU_scores.txt', 'w') as f:
        for s in scores:
            f.write("%s\n" % s)
    
    print('File saved.')

    
