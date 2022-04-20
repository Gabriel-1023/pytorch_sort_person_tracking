"""
This code is used to batch detect images in a folder.
"""
import argparse
import os
import sys
import time
import numpy
import cv2

from sort.sort import Sort
from vision.ssd.config.fd_config import define_img_size


def draw_bboxes(image, bboxes, line_thickness):
    line_thickness = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) * 0.5) + 1

    list_pts = []
    point_radius = 4

    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        color = (0, 255, 0)

        # 撞线的点
        check_point_x = x1
        check_point_y = int(y1 + ((y2 - y1) * 0.6))

        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)

        font_thickness = max(line_thickness - 1, 1)
        t_size = cv2.getTextSize(cls_id, 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, line_thickness / 3,
                    [225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)

        list_pts.append([check_point_x - point_radius, check_point_y - point_radius])
        list_pts.append([check_point_x - point_radius, check_point_y + point_radius])
        list_pts.append([check_point_x + point_radius, check_point_y + point_radius])
        list_pts.append([check_point_x + point_radius, check_point_y - point_radius])

        ndarray_pts = numpy.array(list_pts, numpy.int32)

        cv2.fillPoly(image, [ndarray_pts], color=(0, 0, 255))

        list_pts.clear()

    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='detect_imgs')

    parser.add_argument('--net_type', default="RFB", type=str, help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
    parser.add_argument('--input_size', default=320, type=int, help='define network input size,default optional value 128/160/320/480/640/1280')
    parser.add_argument('--threshold', default=0.5, type=float, help='score threshold')
    parser.add_argument('--candidate_size', default=1500, type=int,help='nms candidate size')
    parser.add_argument('--path', default="imgs", type=str,help='imgs dir')
    parser.add_argument('--test_device', default="cuda:0", type=str,help='cuda:0 or cpu')
    args = parser.parse_args()
    define_img_size(args.input_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

    from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
    from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

    result_path = "./detect_imgs_results"
    label_path = "./models/voc-model-labels.txt"
    test_device = args.test_device

    class_names = [name.strip() for name in open(label_path).readlines()]
    if args.net_type == 'slim':
        model_path = "models/pretrained/version-slim-320.pth"
        net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
        predictor = create_mb_tiny_fd_predictor(net, candidate_size=args.candidate_size, device=test_device)
    elif args.net_type == 'RFB':
        model_path = "models/train-version-RFB/RFB-Epoch-299-Loss-2.968528504555042.pth"
        # model_path = "models/pretrained/version-RFB-640.pth"
        net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
        predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=args.candidate_size, device=test_device)
    else:
        print("The net type is wrong!")
        sys.exit(1)
    net.load(model_path)

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    listdir = os.listdir(args.path)

    sort = Sort()
    counter = 0
    resize = (960, 540)
    file_path = "./"
    capture = cv2.VideoCapture('./video/test.mp4')  # './video/test.mp4'
    start_time = time.time()
    while True:
        counter = counter + 1
        _, orig_image = capture.read()
        orig_image = cv2.resize(orig_image, resize)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = predictor.predict(image, args.candidate_size / 2, args.threshold)
        probs = probs.numpy()

        list_bboxes = sort.update(numpy.array(boxes))
        list_bboxes = list_bboxes.astype(int).tolist()
        for i,j in zip(list_bboxes,range(0, len(list_bboxes))):
            i.insert(4, 'person')
            list_bboxes[j] = tuple(list_bboxes[j])

        output_image_frame = draw_bboxes(orig_image, list_bboxes, line_thickness=None)

        if (time.time() - start_time) != 0:  # 实时显示帧数
            cv2.putText(orig_image, "FPS {0}".format(float('%.1f' % (counter / (time.time() - start_time)))), (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imshow('demo', orig_image)
        cv2.waitKey(1)
    capture.release()
    cv2.destroyAllWindows()
    print(sum)
