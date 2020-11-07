import argparse
import cv2
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from utils.util import load_classes, parse_cfg
from models import build_model

# camera shape
CAM_WIDTH, CAM_HEIGHT = 848, 480


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
        transforms.ToTensor(),
        # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])
    img = transform(img)
    img.unsqueeze_(0)

    output = model(img)
    return output


def predict_class_idx_and_confidence(img, model, input_size):
    """predict image class and confidence according to trained model

    Args:
        img: image to predict
        model: pytorch model
    Returns:
        class_: class index
        confidence: class confidence
    """
    # call base prediction function
    output = predict_img(img, model, input_size)
    class_idx = output.max(1)[1]
    confidence = output.max(1)[0]

    # confidence percentage,save three decimal places
    # confidence = '%.2f' % (confidence * 100)

    return class_idx, confidence


classes = load_classes('cfg/classes.cfg')


def predict_class_name_and_confidence(img, model, input_size):
    """predict image class and confidence according to trained model

    Args:
        img: image to predict
        model: pytorch model
    Returns:
        class_name: class index
        confidence: class confidence
    """
    class_idx, confidence = predict_class_idx_and_confidence(
        img, model, input_size)
    class_name = classes[int(class_idx)]

    return class_name, confidence


def predict_and_show_one_img(img, model, input_size):
    """get model output of one image

    Args:
        img: image ndarray
        model: pytorch trained model
    Returns:
        class_name: class name
        confidence: class confidence
    """
    class_name, confidence = predict_class_name_and_confidence(
        img, model, input_size)

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font_text = ImageFont.truetype("data/simsun.ttc", 22, encoding="utf-8")
    draw.text((5, 5), class_name + ' ' + str(confidence) +
              '%', (0, 255, 0), font=font_text)

    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    # cv2.namedWindow('img', 0)
    # cv2.resizeWindow('img',window_width,window_height)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return class_name, confidence


def cv_loop(weight_path, cfg):
    # loop get camera frame and show on window
    model = build_model(weight_path, cfg)
    input_size = int(cfg['input_size'])

    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_WIDTH)
    cap.set(4, CAM_HEIGHT)

    # main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Camera frame load failed!')
            break
        print('frame shape:', frame.shape)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        class_name, confidence = predict_class_name_and_confidence(
            img, model, input_size)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        font_text = ImageFont.truetype("data/simsun.ttc", 26, encoding="utf-8")
        draw.text((5, 5), class_name + ' ' +
                  str(confidence)+'%', (0, 255, 0), font=font_text)
        frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        print('Class name:', class_name, 'Confidence:', str(confidence)+'%')
        # cv2.namedWindow('frame', 0)
        cv2.resizeWindow('frame', (int(cap.get(3)), int(cap.get(4))))
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='Food Recognition System')

    # parser.add_argument("--image", "-i", dest='image', default='data/samples/test.jpg',
    #                     help="Path of image to perform detection upon", type=str)

    # parser.add_argument("--video", "-v", dest='video', default='data/samples/test.mp4',
    #                     help="Path of video to run detection upon", type=str)

    parser.add_argument("--weight", "-w", dest='weight', default="weights/frs_cnn.pth",
                        help="Path of model weight", type=str)

    parser.add_argument("--source", "-s", dest='source', default="data/samples/test.jpg",
                        help="Path of your input file source,0 for webcam", type=str)

    parser.add_argument("--cfg", "-c", dest='cfg', default="cfg/frs.cfg",
                        help="Your config file path", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    weight_path, cfg_path, source = args.weight, args.cfg, args.source

    cfg = parse_cfg(cfg_path)
    input_size = int(cfg['input_size'])

    model = build_model(weight_path, cfg)
    print('Model successfully loaded!')

    if source.split('.')[-1] in ['jpg', 'png', 'jpeg', 'bmp', 'tif', 'tiff', 'gif']:
        img = cv2.imread(source)
        class_name, confidence = predict_and_show_one_img(
            img, model, input_size)
        print('Class name:', class_name, 'Confidence:', str(confidence)+'%')
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
                "data/simsun.ttc", 26, encoding="utf-8")
            draw.text((5, 5), class_name+' ' + str(confidence) +
                      '%', (0, 255, 0), font=font_text)
            frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

            print('Class name:', class_name, 'Confidence:', str(confidence)+'%')
            cv2.resizeWindow('frame', (int(cap.get(3)), int(cap.get(4))))
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    elif source == '0':
        cap = cv2.VideoCapture(0)
        cap.set(3, CAM_WIDTH)
        cap.set(4, CAM_HEIGHT)
        # main loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print('Camera frame load failed!')
                break
            print('frame shape:', frame.shape)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            class_name, confidence = predict_class_name_and_confidence(
                img, model, input_size)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)
            font_text = ImageFont.truetype(
                "data/simsun.ttc", 26, encoding="utf-8")
            draw.text((5, 5), class_name + ' ' +
                      str(confidence)+'%', (0, 255, 0), font=font_text)
            frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

            print('Class name:', class_name, 'Confidence:', str(confidence)+'%')
            cv2.resizeWindow('frame', (int(cap.get(3)), int(cap.get(4))))
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print('your --source value not correct!')
