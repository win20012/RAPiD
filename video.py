from api import Detector
import numpy as np
import cv2

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
def cv2pil(imgCV):
    imgCV_RGB = cv2.cvtColor(imgCV, cv2.COLOR_BGR2RGB)
    imgPIL = Image.fromarray(imgCV_RGB)
    return imgPIL  

detector = Detector(model_name='rapid',
                    weights_path='./weights/pL1_MWHB1024_Mar11_4000.ckpt',use_cuda=True)

cap = cv2.VideoCapture('videos/testvid.mp4')

while(cap.isOpened()):
    # フレームを取得
    ret, frame = cap.read()

    if ret == False:
        break
    #pil_img1=cv2pil(frame)
    #npobj=detector(model_name='rapid',)
    np_img = detector.detect_one(pil_img=cv2pil(frame),
                    input_size=1024, conf_thres=0.3,
                    return_img=True)
    new_image = np.array(np_img, dtype=np.uint8)
    np_img2 = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    # フレームを表示
    cv2.imshow("Frame", np_img2)

    # qキーが押されたら途中終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('finished')