import cv2
from viewer import AndroidViewer
import time, threading
from zmq_client import ZmqClient
from read_mp4 import cal_offset
import json
import uvicorn
import math
import traceback
# This will deploy and run server on android device connected to USB
android = AndroidViewer(max_width=720, bitrate=800000,port=8081)

x, y = 354, 187
cancel_x, cancel_y = 644, 81
currentFrame = None

def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=1):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def checkHasRed(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(img_hsv, (0,50,20), (5,255,255))
    mask2 = cv2.inRange(img_hsv, (175,50,20), (180,255,255))
    mask = cv2.bitwise_or(mask1, mask2 )
    # croped = cv2.bitwise_and(img, img, mask=mask)
    # Determine if the color exists on the image
    count_red = cv2.countNonZero(mask)
    if count_red > 0:
        return True, count_red
    else:
        return False, count_red


def shooting(blood, start_x, start_y, end_y, width, height):
    ######################################################### search new location
    global currentFrame
    new_frame = currentFrame.copy()
    new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    blood_gray = cv2.cvtColor(blood, cv2.COLOR_BGR2GRAY)
    
    method = cv2.TM_SQDIFF_NORMED
    
    # bounding_size = 100
    # new_frame_gray = new_frame_gray[max(start_y - bounding_size, 0):min(end_y + bounding_size, height - 1), max(start_x - bounding_size, 0):min(end_x + bounding_size, width - 1)]
    res = cv2.matchTemplate(new_frame_gray, blood_gray, method=method)
    # res = cv2.matchTemplate(new_frame, blood, method=method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    # w, h = blood_gray.shape[::-1]
    bottom_right = (top_left[0] + width, top_left[1] + height)
    template_threshold = 35
    template_min = 4
    
    abs_y = abs(top_left[1] - start_y)
    abs_x = abs(top_left[0] - start_x)
    print('abs_x', top_left[0] - start_x, 'abs_y', top_left[1] - start_y)
    
    # cv2.imwrite('blood1.png', frame[start_y:end_y, start_x:end_x])
    # cv2.imwrite('blood2.png', new_frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]])

    if abs_x < template_threshold and abs_y < template_threshold and abs_y > 0:
        print('---->', 'origin({}, {})'.format(start_x, start_y), 'after({}, {})'.format(top_left[0], top_left[1])) 
        
        scale_moving_x = 1.8/ (abs_y + abs_x) * abs_x
        scale_moving_y = 1.8/ (abs_y + abs_x) * abs_y
        
        start_x = start_x + (top_left[0] - start_x) * scale_moving_x
        start_y = start_y + (top_left[1] - start_y) * scale_moving_y
        print('modify x', start_x, 'modify y', start_y)

    ######################################################### search new location

    abs_y = abs(y - end_y + 20)
    print('--y', y, end_y + 20, abs_y)
    # if abs_y < 30:
    #     enermy = [int(start_x + width / 2), end_y + 10]
    # elif abs_y < 78:
    #     enermy = [int(start_x + width / 2), end_y + 38]
    # else:
    enermy = [int(start_x + width / 2), end_y + 15]
        
    abs_x = abs(x - enermy[0])
    print('--x',x, enermy[0], abs_x)
    if abs_x > 110 and abs_y > 20:
        if x < enermy[0] and y < enermy[1]:
            if abs_y < 60:
                enermy = [int(start_x + width / 2), end_y + 50]
            else:
                enermy = [int(start_x + width / 2), end_y + 80]
        # else:
            # enermy = [int(start_x + width / 2), end_y + 5]
    elif abs_x < 110 and abs_y < 60:
        enermy = [int(start_x + width / 2), end_y + 45]
    elif abs_x > 110 and abs_y < 20:
        enermy = [int(start_x + width / 2), end_y + 40]
    elif abs_x < 110 and abs_y > 100:
        if y < enermy[1]:
            enermy = [int(start_x + width / 2), end_y + 65]
        else:
            enermy = [int(start_x + width / 2), end_y + 5]
        
    
    ## baili center
    base_center = (344, 187)
    c2 = (enermy[0]+10, enermy[1] + 10)
    cv2.rectangle(new_frame, (enermy[0], enermy[1]), c2, color=(128, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.rectangle(new_frame, base_center, (base_center[0]+5, base_center[1] +5), color=(128, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    # plot_one_box(xyxy, new_frame, label=label)
    cv2.imwrite('frame.png', new_frame)
    return enermy

def getOffsetFromFrame(frame, threshold):
    enermies = client.send_cvimg(frame)
    if len(enermies) == 0:
        return None, None
    enermies = sorted(enermies, key=lambda k: k['conf'], reverse=True)
    enermy_blood = []
    for resp in enermies:
        label = resp['label']
        xyxy = resp['pos']
        conf = resp['conf']
        if float(conf) > threshold:
            start_x, start_y, end_x, end_y = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            
            blood = frame[start_y:end_y, start_x:end_x]
            
            # Determine if the color exists on the image
            ret, count_red = checkHasRed(blood)
            if ret:
                # print('Red is present!')
                pass
            else:
                # plot_one_box(xyxy, frame, label=label)
                # cv2.imwrite('no-frame.png', frame)
                continue
            width, height = end_x - start_x, end_y - start_y
            
            # print(width, height,'----')
            if width <= 45 or height <= 10:
                # not enermy
                continue
            
            distance = math.sqrt(math.pow((x - start_x), 2) + math.pow((y-start_y), 2))
            
            enermy_blood.append({'start_x': start_x, 'start_y': start_y, 'end_x': end_x, 'end_y': end_y, 'width': width, 'height': height, 'count_red': count_red, 'blood': blood, 'distance': distance})
    if len(enermy_blood) == 0:
        return None, None
    # for
    enermy_blood = sorted(enermy_blood, key=lambda k: k['distance'])
    shoot_target = enermy_blood[0]
    if len(enermy_blood) >= 2:
        shoot_target_1 = enermy_blood[1]
        if abs(shoot_target['distance'] - shoot_target_1['distance']) <= 100 and shoot_target_1['count_red'] < shoot_target['count_red']:
            shoot_target = shoot_target_1
    start_x = shoot_target['start_x']
    start_y = shoot_target['start_y']
    end_x = shoot_target['end_x']
    end_y = shoot_target['end_y']
    width = shoot_target['width']
    height = shoot_target['height']
    blood = shoot_target['blood']
    enermy = shooting(blood, start_x, start_y, end_y, width, height)
    return enermy[0], enermy[1]


def cature_screen(android):
    print('start', 'cature_screen')
    while True:
        # time.sleep(1)
        frames = android.get_next_frames()
        if frames is None:
            continue
        
        for frame in frames:
            global currentFrame
            currentFrame = frame.copy()
            # cv2.imwrite('current.png', frame)
            
            # cv2.imshow('game', frame)
            # if cv2.waitKey(1) & 0xFF == ord('s'):
            #     cv2.imwrite('val/{}.png'.format(index), frame)
            #     print(frame.shape)
            #     index += 1


def create_fastapi_app():
    from fastapi import FastAPI, Request
    # from flask_json import JsonError

    app = FastAPI()


    @app.get('/shoot')
    def predict(request: Request):
        global android
        # android.touch_down(x, y, 1.8)
        android.touch_down(x, y, 0.2)
        # TODO: tune this value
        # x - 30
        android.touch_move(x, y + 30, 1.5) # 1.4  origin 1.6  1.5 better
        isTouch = True
        try:
            frame = currentFrame.copy()
            x_offset, y_offset = getOffsetFromFrame(frame, threshold)
            if x_offset is not None and y_offset is not None:
                # android.touch_move(x_offset, y_offset, 0.5)
            
                android.touch_up(x_offset, y_offset)
                # android.swipe(x, y, x_offset, y_offset, with_down=False)
                # new_frame = currentFrame.copy()
                # cv2.imwrite('frame_shoot.png', new_frame)
                isTouch = False

        except Exception as e:
            traceback.print_exc()
        if isTouch:
            # android.touch_move(x-30, y + 30)
            # android.touch_move(cancel_x, cancel_y)
            android.swipe(cancel_x, cancel_y - 60, cancel_x, cancel_y, with_down=False)
        # cancel touch
        android.touch_up(cancel_x, cancel_y)
        return 0

    return app

client = ZmqClient()

index = 60

threshold = 0.8


# swipe = threading.Thread(target=swipe_screen, args=(android,))
# swipe.start()

cature = threading.Thread(target=cature_screen, args=(android,))
cature.start()

app = create_fastapi_app()
uvicorn.run(app, host="0.0.0.0", port=8080)
