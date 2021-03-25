import argparse
import os
import shutil

import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from models import build_model
from utils import load_classes, parse_cfg

parser = argparse.ArgumentParser(description='Food Recognition System')
parser.add_argument("--weights", "-w", help="Path of model weight", type=str, default="weights/frs_cnn.pth")
parser.add_argument("--source", "-s", help="Path of your input file source, 0 for webcam", type=str,
                    default="data/samples/test.jpg")
parser.add_argument('--output', "-o", help='Output folder', type=str, default='output')
parser.add_argument("--cfg", "-c", help="Your config file path", type=str, default="cfg/frs.cfg")
parser.add_argument("--input_size", "-i", help="Image input size", type=int, default=224)
parser.add_argument("--cam_width", "-cw", help="Camera width", type=int, default=848)
parser.add_argument("--cam_height", "-ch", help="Camera height", type=int, default=480)
parser.add_argument("--font_type", "-ft", help="Path of font type", type=str, default="data/simsun.ttc")
args = parser.parse_args()


def predict_img(img, model, input_size):
    """get model prediction of one image

    Args:
        img: image ndarray
        model: pytorch trained model
    Returns:
        output: pytorch model output
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
        # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])
    img = transform(img)
    img.unsqueeze_(0)

    output = model(img)
    output = F.softmax(output, dim=1)
    return output


def predict_class_idx_and_confidence(img, model, input_size):
    """predict image class and confidence according to trained model

    Args:
        img: image to predict
        model: pytorch model
    Returns:
        class_: class index
        confidence: class confidence 0~1
    """
    # call base prediction function
    output = predict_img(img, model, input_size)
    confidence = output.max(1)[0].item()
    class_idx = output.max(1)[1].item()

    return class_idx, confidence


classes = load_classes('cfg/classes.cfg')


def predict_class_name_and_confidence(img, model, input_size):
    """predict image class and confidence according to trained model

    Args:
        img: image to predict
        model: pytorch model
        input_size: image input size
    Returns:
        class_name: class index
        confidence: class confidence
    """
    class_idx, confidence = predict_class_idx_and_confidence(
        img, model, input_size)
    class_name = classes[int(class_idx)]

    return class_name, confidence


def predict_and_show_img(img, model, input_size):
    """get model output of one image and show

    Args:
        img: image ndarray
        model: pytorch trained model
        input_size: image input size
    Returns:
        class_name: class name
        confidence: class confidence
        img: predicted and drawn img
    """
    class_name, confidence = predict_class_name_and_confidence(
        img, model, input_size)

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font_text = ImageFont.truetype(args.font_type, 22, encoding="utf-8")
    draw.text((5, 5), class_name + ' ' + str('%.2f' % (confidence * 100)) + '%', (0, 255, 0), font=font_text)

    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return class_name, confidence, img


def predict_and_draw_img(img, model, input_size):
    """get model output of one image and draw

    Args:
        img: image ndarray
        model: pytorch trained model
        input_size: image input size
    Returns:
        class_name: class name
        confidence: class confidence
        img: predicted img
    """
    class_name, confidence = predict_class_name_and_confidence(
        img, model, input_size)

    # draw predict
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font_text = ImageFont.truetype(args.font_type, 22, encoding="utf-8")
    draw.text((5, 5), class_name + ' ' + str(confidence) +
              '%', (0, 255, 0), font=font_text)

    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    return class_name, confidence, img


def cv_loop(weight_path, cfg, input_size):
    # loop get camera frame and show on window
    model = build_model(weight_path, cfg)

    cap = cv2.VideoCapture(0)
    cap.set(3, args.cam_width)
    cap.set(4, args.cam_height)

    # main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Camera frame load failed!')
            break
        print('Frame shape:', frame.shape)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        class_name, confidence = predict_class_name_and_confidence(
            img, model, input_size)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        font_text = ImageFont.truetype(args.font_type, 26, encoding="utf-8")
        draw.text((5, 5), class_name + ' ' +
                  str('%.2f' % (confidence * 100)) + '%', (0, 255, 0), font=font_text)
        frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        print('Class name:', class_name, 'Confidence:', str('%.2f' % (confidence * 100)) + '%')
        # cv2.namedWindow('frame', 0)
        cv2.resizeWindow('Frame', (int(cap.get(3)), int(cap.get(4))))
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    weight_path, cfg_path, source, output, input_size = args.weights, args.cfg, args.source, args.output, args.input_size

    # load configs from file
    cfg = parse_cfg(cfg_path)

    # load model
    model = build_model(weight_path, cfg)
    print('Model successfully loaded!')

    # create output dir
    if not os.path.exists(output):
        os.makedirs(output)

    # image
    if source.split('.')[-1] in ['jpg', 'png', 'jpeg', 'bmp', 'tif', 'tiff', 'gif']:
        img = cv2.imread(source)
        img_name = os.path.basename(source)
        class_name, confidence, img = predict_and_show_img(
            img, model, input_size)

        # save output img
        print(os.path.join(output, source))
        cv2.imwrite(os.path.join(output, img_name), img)
        print('Class name:', class_name, 'Confidence:', str('%.2f' % (confidence * 100)) + '%')

    # video
    elif source.split('.')[-1] in ['mp4', 'avi', 'mkv', 'flv', 'rmvb', 'mov', 'rm']:
        cap = cv2.VideoCapture(source)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print('Video load failed!')
                break

            class_name, confidence = predict_class_name_and_confidence(
                frame, model, input_size)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)
            font_text = ImageFont.truetype(
                args.font_type, 26, encoding="utf-8")
            draw.text((5, 5), class_name + ' ' + str('%.2f' % (confidence * 100)) + '%', (0, 255, 0), font=font_text)
            frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

            print('Class name:', class_name, 'Confidence:', str('%.2f' % (confidence * 100)) + '%')
            cv2.resizeWindow('Frame', (int(cap.get(3)), int(cap.get(4))))
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # webcam
    elif source == '0':
        cap = cv2.VideoCapture(0)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)
        if not cap.isOpened():
            print('Camera not open!')
        # main loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print('Camera frame load failed!')
                break
            print('Frame shape:', frame.shape)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            class_name, confidence = predict_class_name_and_confidence(
                img, model, input_size)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)
            font_text = ImageFont.truetype(
                args.font_type, 26, encoding="utf-8")
            draw.text((5, 5), class_name + ' ' +
                      str('%.2f' % (confidence * 100)) + '%', (0, 255, 0), font=font_text)
            frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

            print('Class name:', class_name, 'Confidence:', str('%.2f' % (confidence * 100)) + '%')
            cv2.resizeWindow('Frame', (int(cap.get(3)), int(cap.get(4))))
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # folder
    elif source == source.split('.')[-1]:
        # create output folder
        output = os.path.join(output, source.split('/')[-1])
        if os.path.exists(output):
            shutil.rmtree(output)
            # os.removedirs(output)
        os.makedirs(output)

        imgs = os.listdir(source)
        for img_name in imgs:
            # img = cv2.imread(os.path.join(source, img_name))
            img = cv2.imdecode(np.fromfile(os.path.join(
                source, img_name), dtype=np.uint8), cv2.IMREAD_COLOR)
            class_name, confidence, img = predict_and_draw_img(
                img, model, input_size)
            print(img_name)
            print('Class name:', class_name, 'Confidence:', str('%.2f' % (confidence * 100)) + '%')
            # save output img
            cv2.imwrite(os.path.join(output, img_name), img)
