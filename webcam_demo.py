import cv2
import argparse
import glob
import numpy as np
import os
import time
import sys

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
from tracker import re3_tracker

from utils import drawing
from utils import bb_util
from utils import im_util

from constants import OUTPUT_WIDTH
from constants import OUTPUT_HEIGHT
from constants import PADDING

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

drawnBox = np.zeros(4)
boxToDraw = [np.zeros(4)]
mousedown = False
mouseupdown = False
initialize = [False]
cnt_obj = 0
colors = [list(np.random.random(size=3) * 256)]

def on_mouse(event, x, y, flags, params):
    global mousedown, mouseupdown, drawnBox, boxToDraw, initialize, cnt_obj
    if event == cv2.EVENT_LBUTTONDOWN:
        drawnBox[[0,2]] = x
        drawnBox[[1,3]] = y
        mousedown = True
        mouseupdown = False
    elif mousedown and event == cv2.EVENT_MOUSEMOVE:
        drawnBox[2] = x
        drawnBox[3] = y
    elif event == cv2.EVENT_LBUTTONUP:
        drawnBox[2] = x
        drawnBox[3] = y
        mousedown = False
        mouseupdown = True
        initialize[cnt_obj] = True
    boxToDraw[cnt_obj] = drawnBox.copy()
    boxToDraw[cnt_obj][[0,2]] = np.sort(boxToDraw[cnt_obj][[0,2]])
    boxToDraw[cnt_obj][[1,3]] = np.sort(boxToDraw[cnt_obj][[1,3]])
    if event == cv2.EVENT_LBUTTONUP:
        boxToDraw.append(np.zeros(4))
        cnt_obj += 1
        initialize.append(False)
        colors.append(list(np.random.random(size=3) * 256))


def show_webcam(mirror=False):
    global tracker, initialize
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Webcam', OUTPUT_WIDTH, OUTPUT_HEIGHT)
    cv2.setMouseCallback('Webcam', on_mouse, 0)
    frameNum = 0
    # outputDir = None
    outputBoxToDraw = None
    if RECORD:
        # print('saving')
        if not os.path.exists('outputs'):
            os.mkdir('outputs')
        tt = time.localtime()
        # outputDir = ('outputs/%02d_%02d_%02d_%02d_%02d/' %
        #         (tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))
        # os.mkdir(outputDir)
        # labels = open(outputDir + 'labels.txt', 'w')
        video_writer = cv2.VideoWriter('outputs/%02d_%02d_%02d_%02d_%02d.avi' %
                (tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec),
                cv2.VideoWriter_fourcc(*'DIVX'), 10, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    while True:
        _, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        # origImg = img.copy()
        if mousedown:
            for i in range(cnt_obj+1):
                cv2.rectangle(img,
                        (int(boxToDraw[i][0]), int(boxToDraw[i][1])),
                        (int(boxToDraw[i][2]), int(boxToDraw[i][3])),
                        colors[i], PADDING)
                cv2.putText(img, 'Object %02d' % i,
                            (int(boxToDraw[i][0]), int(boxToDraw[i][1]-5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i])
            if RECORD:
                cv2.circle(img, (int(drawnBox[2]), int(drawnBox[3])), 10, [255,0,0], 4)
        elif mouseupdown:
            for i in range(cnt_obj):
                if initialize[i]:
                    outputBoxToDraw = tracker.track(i, img[:,:,::-1], boxToDraw[i])
                    initialize[i] = False
                else:
                    # begin_time = time.time()
                    outputBoxToDraw = tracker.track(i, img[:,:,::-1])
                    boxToDraw[i] = outputBoxToDraw
                    # print('fps: ', time.time()-begin_time)
                cv2.rectangle(img,
                        (int(outputBoxToDraw[0]), int(outputBoxToDraw[1])),
                        (int(outputBoxToDraw[2]), int(outputBoxToDraw[3])),
                        colors[i], PADDING)
                cv2.putText(img, 'Object %02d' % i,
                            (int(boxToDraw[i][0]), int(boxToDraw[i][1]-5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i])
        cv2.imshow('Webcam', img)
        if RECORD:
            # if outputBoxToDraw is not None:
            #     labels.write('%d %.2f %.2f %.2f %.2f\n' %
            #             (frameNum, outputBoxToDraw[0], outputBoxToDraw[1],
            #                 outputBoxToDraw[2], outputBoxToDraw[3]))
            # cv2.imwrite('%s%08d.jpg' % (outputDir, frameNum), origImg)
            # print('saving')
            video_writer.write(img)

        keyPressed = cv2.waitKey(1)
        if keyPressed == 27 or keyPressed == 1048603:
            break  # esc to quit
        frameNum += 1
    cv2.destroyAllWindows()
    if RECORD:
        video_writer.release()

# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Show the Webcam demo.')
    parser.add_argument('-r', '--record', action='store_true', default=False)
    args = parser.parse_args()
    RECORD = args.record


    tracker = re3_tracker.Re3Tracker()

    show_webcam(mirror=True)


