import datetime
from threading import Thread

import torch
from torch.backends import cudnn
from torchvision.transforms import GaussianBlur

from deep_sort.deep_sort import DeepSort
import cv2
from pathlib import Path
import os

import sys

sys.path.insert(0, './yolov5')
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config

from stream.inout_saving import save_inout_data

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
count1 = 0  # 인원수 카운트
counting_obj1 = set()  # 한번 카운트 하면 또 안하게
data1 = set()
count2 = 0  # 인원수 카운트
counting_obj2 = set()
data2 = set()

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# DEVICE1 = select_device('')
# MODEL1 = DetectMultiBackend('ai.pt', device=DEVICE1, dnn=False)
# STRIDE1, NAMES1, PT1, JIT1, _ = MODEL1.stride, MODEL1.names, MODEL1.pt, MODEL1.jit, MODEL1.onnx
# IMG_SIZE1 = check_img_size([640, 640], s=STRIDE1)  # check image size
# ip_camera1 = LoadStreams('https://www.youtube.com/watch?v=BBdC1rl5sKY', img_size=IMG_SIZE1, stride=STRIDE1, auto=PT1 and not JIT1)


class Streaming:
    def __init__(self, source, is_in):
        self.stream = None

    def get_stream(self):
        return self.stream

    def streaming(self, source, is_in):
        yolo_model = 'ai.pt'
        deep_sort_model = 'osnet_x0_25'

        image_size = [640, 640]
        conf_thres = 0.3
        iou_thres = 0.5
        device = ''
        classes = None
        agnostic_nms = False
        augment = False
        deepsort_config = 'deep_sort/configs/deep_sort.yaml'
        half = False
        visualize = False
        max_det = 1000
        dnn = False

        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file(deepsort_config)
        deepsort = DeepSort(deep_sort_model,
                            max_dist=cfg.DEEPSORT.MAX_DIST,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

        webcam = source == '0' or source.startswith(
            'rtsp') or source.startswith('http') or source.endswith('.txt')

        # Initialize
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(yolo_model, device=device, dnn=dnn)
        stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
        image_size = check_img_size(image_size, s=stride)  # check image size

        # Half
        half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.model.half() if half else model.model.float()

        if webcam:
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=image_size, stride=stride, auto=pt and not jit)
        else:
            dataset = LoadImages(source, img_size=image_size, stride=stride, auto=pt and not jit)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *image_size).to(device).type_as(next(model.model.parameters())))  # warmup

        dt, seen = [0.0, 0.0, 0.0, 0.0], 0
        for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
            t1 = time_sync()
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, _ = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

                s += '%gx%g ' % img.shape[2:]  # print string

                annotator = Annotator(im0, line_width=2, pil=not ascii)
                w, h = im0.shape[1], im0.shape[0]
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    # pass detections to deepsort
                    t4 = time_sync()
                    outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                    t5 = time_sync()
                    dt[3] += t5 - t4

                    # draw boxes for visualization
                    if len(outputs) > 0:
                        for j, (output, conf) in enumerate(zip(outputs, confs)):

                            bboxes = output[0:4]  # 박스 꼭짓점 들 인듯?
                            id = output[4]  # 아마도 객체 아이디
                            cls = output[5]  # class 이다. 욜로 프리트레인모델 기준 0은 person
                            #######################
                            # count
                            if is_in:
                                count_obj_in(bboxes, w, h, id, int(cls))
                            else:
                                count_obj_out(bboxes, w, h, id, int(cls))
                            ########################

                            c = int(cls)  # integer class
                            if c == 0:
                                label = f'{id} {names[c]} {conf:.2f}'
                                annotator.box_label(bboxes, label, color=colors(c, True))

                                ##############
                                # blur
                                # print('bbox', bboxes)
                                # blurring = GaussianBlur((7, 13), sigma=(0.1, 0.2))
                                # # blur_img = blurring(torch.tensor(bboxes))
                                # blur_img = blurring(torch.tensor(bboxes))
                                # cv2.imshow(str(p), blur_img)
                                ##############

                else:
                    deepsort.increment_ages()
                    LOGGER.info('No detections')

                # Stream results
                im0 = annotator.result()

            image_bytes = cv2.imencode('.jpg', im0)[1].tobytes()
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n'


source1 = 'videos/in.mp4'
in_streaming = Streaming(source1, is_in=True)

in_streaming.streaming()

breakfast = False
lunch = False
dinner = False


class People:
    def __init__(self):
        self.count = 0

    def increase(self):
        self.count += 1

    def decrease(self):
        self.count -= 1


people_counter = People()


def count_obj_in(box, w, h, id, cls):
    global count1, data1, counting_obj1, count2, counting_obj2, data2, breakfast, lunch, dinner

    if (w - 800) < int(box[0] + (box[2] - box[0]) / 2) < (w - 400) and int(box[1] + (box[3] - box[1]) / 2) < (
            h - 300) and cls == 0 and id not in data1 and id not in counting_obj1:  # 첫번째에서 인식
        data1.add(id)
    if int(box[0] + (box[2] - box[0]) / 2) < (
            w - 900) and cls == 0 and id in data1 and id not in counting_obj1:  # 한번 확인이 된 아이디이면 카운팅
        count1 += 1
        people_counter.increase()
        counting_obj1.add(id)
    ##########################
    # data saving process
    now = datetime.datetime.now()
    # 7시보다 크고 8시보다 작다면 & breakfast False 라면

    if datetime.datetime(now.year, now.month, now.day, 7, 0, 0) < now < datetime.datetime(now.year, now.month, now.day,
                                                                                          8, 0,
                                                                                          0) and breakfast is False:
        # TODO 한시간 블로킹 개버그
        save_inout_data(0, count1, count2)
        breakfast = True
        count1 = count2 = 0
        counting_obj1 = counting_obj2 = set()
        data1 = data2 = set()
    elif datetime.datetime(now.year, now.month, now.day, 12, 40, 0) < now < datetime.datetime(now.year, now.month,
                                                                                              now.day, 13, 40,
                                                                                              0) and lunch is False:
        save_inout_data(1, count1, count2)
        lunch = True
        count1 = count2 = 0
        counting_obj1 = counting_obj2 = set()
        data1 = data2 = set()
    # 저녁 시간 저장
    elif datetime.datetime(now.year, now.month, now.day, 18, 25, 0) < now < datetime.datetime(now.year, now.month,
                                                                                              now.day, 19, 25,
                                                                                              0) and dinner is False:
        save_inout_data(2, count1, count2)
        dinner = True
        count1 = count2 = 0
        counting_obj1 = counting_obj2 = set()
        data1 = data2 = set()
    # 저녁 시간 이후 리셋
    elif datetime.datetime(now.year, now.month, now.day, 19, 25, 0) < now:
        breakfast = False
        lunch = False
        dinner = False
    ##########################


def count_obj_out(box, w, h, id, cls):
    global count2, counting_obj2, data2
    center_coordinates = (int(box[0] + (box[2] - box[0]) / 2), int(box[1] + (box[3] - box[1]) / 2))  # ( w , h )

    if (w - 800) < center_coordinates[0] < (w - 400) and center_coordinates[1] < (
            h - 400) and cls == 0 and id not in data2 and id not in counting_obj2:  # 첫번째에서 인식
        data2.add(id)
    if center_coordinates[0] < (w - 900) and center_coordinates[1] < (
            h - 300) and cls == 0 and id in data2 and id not in counting_obj2:  # 한번 확인이 된 아이디이면 카운팅
        count2 += 1
        people_counter.decrease()
        counting_obj2.add(id)



