import cv2
import numpy as np
import pyautogui
import time
import sys
import os
import camera

print("start register.py ")

## put text

def put_text(img, point, text):
    cv2.putText(img, text, point, cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('text', img)
    pass


def register():
    text = ["please look at the 'front'                 [   ]",
            "please look at the 'right side of monitor' [   ]",
            "please look at the 'left side of monitor'  [   ]",
            "please look at the 'up side of monitor'   [   ]",
            "please look at the 'down side of monitor'[   ]",
            "ok"]

    img = np.zeros((384, 830, 3), np.uint8)

    name = input("name : ")
    known_path = "knowns/"
    if not os.path.isdir(known_path+name):
        os.mkdir(known_path+name)

    put_text(img, (30, 30), "we're going take a picture")
    cv2.waitKey(2000)

    cam = camera.VideoCamera()

    # register capture img
    write_num = 1
    img_write_path = 'knowns/' + name + '/'
    os.chdir(img_write_path)
    for i in range(5):
        put_text(img, (30, 30+(i+2)*40), text[i])
        cv2.waitKey(2000)
        # add picture
        # 사진 저장 수 count
        count = 1
        while True:
            frame = cam.get_frame()
            cv2.imshow("test", frame)
            if( count % 11 == 0 ):
                break
            time.sleep(0.08)
            cv2.imwrite(str(write_num) + ".png", frame, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])
            print('Saved frame%d.png' % write_num)
            count += 1
            write_num += 1
            # 웹캠 영상 받기 때문
            k = cv2.waitKey(1)
            if k == 27:
                break
            
        put_text(img, (730, 30+(i+2)*40), text[5])
        cv2.waitKey(1000)
        
    put_text(img, (30, 30+8*40), "thank you")
    cv2.waitKey(2000)

    cv2.destroyAllWindows()

# make dataset
def make_dataset():
    pass
    



# train embedding

register()
make_dataset()