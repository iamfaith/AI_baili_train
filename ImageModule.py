import aicmder as cmder
from aicmder.module.module import serving, moduleinfo
import io
from PIL import Image
import json
import base64
import cv2
import numpy as np

import torch
from utils.torch_utils import select_device, load_classifier, time_sync
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import colors, plot_one_box

def readb64(base64_string, save=False):
    # sbuf = StringIO()
    # sbuf.write(base64.b64decode(base64_string))
    # pimg = Image.open(sbuf)
    img_array = io.BytesIO(base64.b64decode(base64_string))
    pimg = Image.open(img_array) # RGB
    if save:
        pimg.save('image.png', 'PNG')
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR) #BGR

@moduleinfo(name='image')
class ImagePredictor(cmder.Module):
    
    def __init__(self, file_path, **kwargs) -> None:
        print('init', file_path)
        self.device = select_device('')
        weights = ["/home/faith/android_viewer/thirdparty/yolov5/runs/train/exp2/weights/best.pt"]
        self.model = attempt_load(weights, map_location=self.device)
        self.stride = int(self.model.stride.max())  # model stride
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names

        self.imgsz = 768
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
    
    # json base64
    @serving
    def predict(self, img_base64):
        print('receive')
        try:
            img0 = readb64(img_base64) #BGR
            # Padded resize
            img = letterbox(img0, self.imgsz, stride=self.stride)[0]
            # Convert
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(self.device)
            img = img.float()  # uint8 to fp16/32
            
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            
            # Inference
            t1 = time_sync()
            visualize, augment = False, False
            pred = self.model(img, augment=augment, visualize=visualize)[0]

            conf_thres, iou_thres, classes, agnostic_nms, max_det= 0.25, 0.45, None, False, 1000
            line_thickness = 3
            hide_labels = False
            hide_conf = False
            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            t2 = time_sync()
            print(f'Done. ({t2 - t1:.3f}s)')
            result = []
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
                        pos = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                        result.append({'label': label, 'pos': pos, 'conf': f'{conf:.2f}'})
            result_json = json.dumps(result)
            print(result_json)
            return result_json
            ####################################################  write result
            # for i, det in enumerate(pred):  # detections per image

            #     s = '%gx%g ' % img.shape[2:]  # print string
            #     if len(det):
            #         # Rescale boxes from img_size to im0 size
            #         det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                
            #         # Print results
            #         for c in det[:, -1].unique():
            #             n = (det[:, -1] == c).sum()  # detections per class
            #             s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

            #         # Write results
            #         for *xyxy, conf, cls in reversed(det):
                        
            #             c = int(cls)  # integer class
            #             label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
            #             plot_one_box(xyxy, img0, label=label, color=colors(c, True), line_thickness=line_thickness)

            #     # Print time (inference + NMS)
            #     print(f'{s}Done. ({t2 - t1:.3f}s)')
                
            # cv2.imwrite('detected.png', img0)
            ####################################################  write result

            # cv2.imshow('after', img0)
            # cv2.waitKey(1)  # 1 millisecond

            # cv2.imshow('base64', img)
            # cv2.waitKey(1)
        except Exception as e:
            print(e)
            return "OK"
    
