# @Author: Ivan
# @LastEdit: 2020/9/25
import os
import argparse
import cv2  # install
import torch
from torchvision import datasets, transforms, utils
from PIL import Image, ImageDraw, ImageFont  # install
import numpy as np  # install
from models import Net
from util import load_classes, load_prices, load_pytorch_model

# input shape
width, height = 100, 100


def predict_img(img, model):
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
        transforms.Resize((width, height)),
        # transforms.RandomRotation(10),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])
    img = transform(img)
    img.unsqueeze_(0)

    output = model(img)
    return output


def predict_class_idx_and_confidence(img, model):
    """predict image class and confidence according to trained model

    Args:
        img: image to predict
        model: pytorch model
    Returns:
        class_: class index
        confidence: class confidence
    """
    # call base prediction function
    output = predict_img(img, model)
    class_idx = output.max(1)[1]
    confidence = output.max(1)[0]

    # confidence percentage,save three decimal places
    # confidence = '%.2f' % (confidence * 100)

    return class_idx, confidence


classes = load_classes('cfg/classes.cfg')


def predict_class_name_and_confidence(img, model):
    """predict image class and confidence according to trained model

    Args:
        img: image to predict
        model: pytorch model
    Returns:
        class_name: class index
        confidence: class confidence
    """
    class_idx, confidence = predict_class_idx_and_confidence(img, model)
    class_name = classes[int(class_idx)]

    return class_name, confidence


def predict_and_show_one_img(img, model):
    """get model output of one image

    Args:
        img: image ndarray
        model: pytorch trained model
    Returns:
        class_name: class name
        confidence: class confidence
    """
    class_name, confidence = predict_class_name_and_confidence(img, model)

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font_text = ImageFont.truetype("simsun.ttc", 26, encoding="utf-8")
    draw.text((5, 5), class_name, (0, 255, 0), font=font_text)

    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    # cv2.namedWindow('img', 0)
    # cv2.resizeWindow('img',window_width,window_height)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return class_name, confidence


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='Food Recognition System')

    parser.add_argument("--image", "-i", dest='image',
                        help="Path of image to perform detection upon", type=str)

    parser.add_argument("--video", "-v", dest='video',
                        help="Path of video to run detection upon", type=str)

    parser.add_argument("--model", "-m", dest='model', help="Path of network model",
                        default="frs_cnn.pth", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    image_path = args.image
    video_path = args.video
    model_path = args.model

    model = load_pytorch_model(model_path)
    print('Model successfully loaded...')

    if image_path:
        # img = cv2.imread(image_path)
        # predict_and_show_one_img(img,model)
        img = cv2.imread(image_path)
        class_name, confidence = predict_class_name_and_confidence(img, model)
        print('Class name:', class_name)
    elif video_path:
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            class_name, confidence = predict_class_name_and_confidence(
                frame, model)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)
            font_text = ImageFont.truetype("simsun.ttc", 26, encoding="utf-8")
            draw.text((5, 5), class_name, (0, 255, 0), font=font_text)
            frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

            print('Class name:', class_name)
            cv2.resizeWindow('frame', (int(cap.get(3)), int(cap.get(4))))
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
