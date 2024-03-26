from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.utils import DEFAULT_CFG
from ultralytics import YOLO
import cv2
import torch
import numpy as np
from tqdm import tqdm

# 载入模型
model = 'best.pt'
# 添加预测源
source = None
# 选择cup还是gpu运行
device = 'cpu'
# 合并参数 verbose设置为False，不单独打印每一帧预测结果
args = dict(model=model, source=source, device=device, save=False, verbose=False)
# 进行预测
predictor = SegmentationPredictor(cfg=DEFAULT_CFG, overrides=args)

# 画图的可视化配置
# 框（rectangle）可视化配置
bbox_color = {
    0: (128, 64, 128),  # road
    1: (244, 35, 232),  # sidewalk
    2: (70, 70, 70),  # building
    3: (102, 102, 156),  # wall
    4: (190, 153, 153),  # fence
    5: (153, 153, 153),  # pole
    6: (250, 170, 30),  # traffic light
    7: (220, 220, 0),  # traffic sign
    8: (107, 142, 35),  # vegetation
    9: (152, 251, 152),  # terrain
    10: (70, 130, 180),  # sky
    11: (220, 20, 60),  # person
    12: (255, 0, 0),  # rider
    13: (0, 0, 142),  # car
    14: (0, 0, 70),  # truck
    15: (0, 60, 100),  # bus
    16: (0, 80, 100),  # train
    17: (0, 0, 230),  # motorcycle
    18: (119, 11, 32),  # bicycle
}  # 框的 BGR 颜色
bbox_thickness = 6  # 框的线宽

# 框类别文字
bbox_labelstr = {
    'font_size': 2,  # 字体大小
    'font_thickness': 2,  # 字体粗细
    'offset_x': 0,  # X 方向，文字偏移距离，向右为正
    'offset_y': -80,  # Y 方向，文字偏移距离，向下为正
}


def predict_frame(img_bgr):
    results = predictor(img_bgr)  # verbose设置为False，不单独打印每一帧预测结果
    # 解析目标检测框
    cls_names = results[0].names  # 类别对应的名称
    bboxes_cls = results[0].boxes.cls.numpy().astype('uint32')  # 检测框所对应的类别ID
    num_bbox = len(bboxes_cls)  # 预测num_bbox框
    bboxes_conf = results[0].boxes.conf.numpy().astype('float32')  # 检测框的置信度
    # 转成整数的 numpy array
    bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')  # 检测框的坐标 xyxy（左上，右下）

    # 解释语义分割结果
    mask = results[0].masks
    # 获取语义分割多个坐标点
    ploy = mask.xy

    # # opencv可视化
    # img_bgr = cv2.imread(source)
    # 创建一个与原始图像同样大小的透明层 进行语义分割填充
    pred_mask_bgr = np.zeros((img_bgr.shape[0], img_bgr.shape[1], 3)).astype('uint8')
    for idx in range(num_bbox):  # 遍历每个框
        # 获取该框坐标
        bbox_xyxy = bboxes_xyxy[idx]

        # 获取框的预测类别
        cls_idx = bboxes_cls[idx]
        bbox_label = cls_names[cls_idx]

        # 获取框的预测置信度
        bbox_conf = bboxes_conf[idx]

        # 画框
        img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]),
                                bbox_color[cls_idx], bbox_thickness)

        # 写框类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
        img_bgr = cv2.putText(img_bgr, bbox_label,
                              (bbox_xyxy[0] + bbox_labelstr['offset_x'], bbox_xyxy[1] + bbox_labelstr['offset_y']),
                              cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color[cls_idx],
                              bbox_labelstr['font_thickness'])

        # 写框置信度
        # img_bgr = cv2.putText(img_bgr, '{:.2f}'.format(bbox_conf), (bbox_xyxy[0]-bbox_labelstr['offset_x'], bbox_xyxy[1]-bbox_labelstr['offset_y']), cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color[cls_idx], bbox_labelstr['font_thickness'])

        pred_mask_bgr = cv2.fillPoly(pred_mask_bgr, [ploy[idx].astype('int32')], color=bbox_color[cls_idx])

    # 混合原始图像和img_mask层
    alpha = 0.4  # 设置透明度
    img_bgr = cv2.addWeighted(pred_mask_bgr, alpha, img_bgr, 1 - alpha, 0)

    return img_bgr


def generate_video(input_path='videos/robot.mp4'):
    filehead = input_path.split('/')[-1].split('.')[0]
    output_path = "out-" + filehead + '.mp4'

    print('视频开始处理', input_path)

    # 获取视频总帧数
    cap = cv2.VideoCapture(input_path)
    frame_count = 0
    while (cap.isOpened()):
        success, frame = cap.read()
        frame_count += 1
        if not success:
            break
    cap.release()
    print('视频总帧数为', frame_count)

    # cv2.namedWindow('Crack Detection and Measurement Video Processing')
    cap = cv2.VideoCapture(input_path)
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, fourcc, fps, (int(frame_size[0]), int(frame_size[1])))

    # 进度条绑定视频总帧数
    with tqdm(total=frame_count - 1) as pbar:
        try:
            while (cap.isOpened()):
                success, frame = cap.read()
                if not success:
                    break
                try:
                    frame = predict_frame(frame)
                except Exception as error:
                    print('报错！', error)
                    pass

                if success == True:
                    # cv2.imshow('Video Processing', frame)
                    out.write(frame)

                    # 进度条更新一帧
                    pbar.update(1)

                # if cv2.waitKey(1) & 0xFF == ord('q'):
                # break
        except:
            print('中途中断')
            pass

    cv2.destroyAllWindows()
    out.release()
    cap.release()
    print('视频已保存', output_path)


generate_video(input_path='stuttgart_00.avi')