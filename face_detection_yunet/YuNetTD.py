import cv2 as cv
import numpy as np
import sys
import os
sys.path.append(r"C:\Users\jleung\Documents\FForFragile\opencv_zoo\models\face_detection_yunet")
from yunet import YuNet

def visualize(image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
    output = image.copy()
    landmark_color = [
        (255,   0,   0), # right eye
        (  0,   0, 255), # left eye
        (  0, 255,   0), # nose tip
        (255,   0, 255), # right mouth corner
        (  0, 255, 255)  # left mouth corner
    ]

    if fps is not None:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    for det in results:
        bbox = det[0:4].astype(np.int32)
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)

        conf = det[-1]
        cv.putText(output, '{:.4f}'.format(conf), (bbox[0], bbox[1]+12), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

        landmarks = det[4:14].astype(np.int32).reshape((5,2))
        for idx, landmark in enumerate(landmarks):
            cv.circle(output, landmark, 2, landmark_color[idx], 2)

    return output


# me - this DAT
# scriptOp - the OP which is cooking
#
# press 'Setup Parameters' in the OP to call this function to re-create the parameters.
def onSetupParameters(scriptOp):
    page = scriptOp.appendCustomPage('Custom')
    #p = page.appendFloat('Valuea', label='Value A')
    #p = page.appendFloat('Valueb', label='Value B')
    return

# called whenever custom pulse parameter is pushed
def onPulse(par):
    return

def onCook(scriptOp):
    MODEL = YuNet(modelPath='face_detection_yunet_2023mar.onnx',
                  inputSize=[320, 320],
                  confThreshold=0.9,
                  nmsThreshold=0.3,
                  topK=5000,
                  backendId=cv.dnn.DNN_BACKEND_OPENCV,
                  targetId=cv.dnn.DNN_TARGET_CPU)
    image = op('meow').numpyArray()
    image = image[:,:,:3]
    image = (image*255).astype(np.uint8)
    image_cv = np.load(open(r"C:\Users\jleung\Documents\FForFragile\opencv_zoo\models\face_detection_yunet\frame.npy","rb"))
    # image_td = np.load(open(r"C:\Users\jleung\Documents\FForFragile\opencv_zoo\models\face_detection_yunet\frame1.npy","rb"))
    image = cv.cvtColor(image,cv.COLOR_RGB2BGR)
    image = cv.rotate(image, cv.ROTATE_180)

    if not os.path.isfile('frame_cv.png'):
        cv.imwrite('frame_cv.png', image_cv)
    if not os.path.isfile('frame_td.png'):
        cv.imwrite('frame_td.png', image)
    # image = cv.resize(image, (640, 480), interpolation = cv.INTER_LINEAR)
    h, w, _ = image.shape
    print(np.min(image),np.max(image))
    print(image.shape)
    
    if not os.path.isfile('frame1.npy'):
        np.save(open('frame1.npy',"wb"), image)
    

    # Inference
    MODEL.setInputSize([w, h])
    
    results = MODEL.infer(image)

    # Print results
    print('{} faces detected.'.format(results.shape[0]))
    for idx, det in enumerate(results):
        print('{}: {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}'.format(
            idx, *det[:-1])
        )

    # Draw results on the input image
    image = visualize(image, results)
    #convert this with opencv to RGBA
    image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    image = cv.cvtColor(image, cv.COLOR_RGB2RGBA)
    image = cv.rotate(image, cv.ROTATE_180)


    scriptOp.copyNumpyArray(image)
    return

