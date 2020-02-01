import tensorflow as tf
import numpy as np
import camera
import cv2
import os

from more import *
import detect_face


class FRTB:
    def __init__(self):
        self.camera = camera.VideoCamera()
        self.ab_file = AboutFile()

        self.registered_names = []
        
        self.location = {'top' : 0, 'bottom' : 0, 'left' : 0, 'right' : 0}

        # initialize registered_face
        
    def __del__(self):
        del self.camera
        
    def get_frame(self, pnet, rnet, onet):
        minsize = 20
        threshold = [0.6, 0.7, 0.7]
        factor = 0.709

        frame = self.camera.get_frame() 
        # for drawing boxes
        frame_with_box = frame[:]

        # find faces
        bounding_boxes, _ = detect_face.detect_face(frame_with_box, minsize, pnet, rnet, onet, threshold, factor)
        print(bounding_boxes)
        num_faces = bounding_boxes.shape[0]
        print("detect face: %d" % num_faces)
        
        # face exist
        if num_faces > 0:
            det = bounding_boxes[:, 0:4]
            img_size = np.asarray(frame_with_box.shape)[0:2]
            bb = np.zeros((num_faces, 4), dtype=np.int32)

            for i in range(num_faces):
                bb[i][0] = det[i][0]
                bb[i][1] = det[i][1]
                bb[i][2] = det[i][2]
                bb[i][3] = det[i][3]
                # location
                self.location['top'] = bb[i][1]
                self.location['bottom'] = bb[i][3]
                self.location['left'] = bb[i][0]
                self.location['right'] = bb[i][2]

                # for location test
                # cv2.putText(frame_with_box, 'top left', (bb[i][0], bb[i][1]), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0,255,255), 3)
                # cv2.putText(frame_with_box, 'top right', (bb[i][2], bb[i][1]), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0,255,255), 3)
                # cv2.putText(frame_with_box, 'bottom left', (bb[i][0], bb[i][3]), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0,255,255), 3)
                # cv2.putText(frame_with_box, 'bottom right', (bb[i][2], bb[i][3]), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0,255,255), 3)

                # 자잘하게 잘못 잡히는 박스 일단 제거
                # 멀리서 얼굴 안 보이면 수정할 것
                garbage_size = 40
                if self.location['right'] - self.location['left'] <  garbage_size:
                    continue
              
                # out of face
                if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                        print('face out')
                        continue
                cv2.rectangle(frame_with_box, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
        else: 
            print("no one is here")

        return frame_with_box


    # button img
    def get_register_img(self):
        img_path = 'button/register_button_img.PNG'
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        return img
    
    def change_register_img(self, img):
        register_img = img[:]
        
        # @@ add register img
        return register_img


# MOUSE

def onMouse(event, x, y, flags, param):
    # global ix, iy
    ix, iy = -1, -1

    knowns_path = 'knowns/'
    register_names = os.listdir(knowns_path)
    if event == cv2.EVENT_LBUTTONDOWN:
        # ix, iy = x, y
        pass
    elif event == cv2.EVENT_MOUSEMOVE:
        pass
    elif event == cv2.EVENT_LBUTTONUP:
        ix, iy = x, y
        
        #register
        if (76 < x < 500) and (483 < y < 624):
            print("registered")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        # delete
        elif (76 < x < 276) and (30 < y < 240):
            print("number1")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        elif (305 < x < 505) and (30 < y < 240):
            print("number2")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        elif (76 < x < 276) and (260 < y < 470):
            print("number3")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        elif (305 < x < 505) and (260 < y < 470):
            print("number4")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        else:
            # others
            pass

        # for print order 
        if ix > x:
            if iy > y:
                print((x, y), (ix, iy))
            else:
                print((x, iy), (ix, y))
        else:
            if iy > y:
                print((ix, y), (x, iy))
            else:
                print((ix, iy), (x, y))



# MAIN

if __name__ == '__main__':

    print("start!")
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        
        frtb = FRTB()
        register_img = frtb.get_register_img()
        cv2.namedWindow("register")
        cv2.setMouseCallback("register", onMouse, param=register_img)

        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, 'src/align')
            ### camera
            

            while True:

                frame = frtb.get_frame(pnet, rnet, onet)
                register_img = frtb.change_register_img(register_img)

                cv2.imshow("Frame", frame)
                cv2.imshow("register", register_img)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            cv2.destroyAllWindows()
            print('finish')
            ### camera end