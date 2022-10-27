import math
import argparse
import time
import datetime
import random
from pathlib import Path
import json
from datasets.hico import make_hico_transforms
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import ImageFolder
import datasets
import util.misc as utils
from datasets import build_dataset
from engine import train_one_epoch, evaluate_hoi
from models import build_model
import os
from PIL import Image
import requests
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'retina'
import numpy as np
import argparse
#import ipywidgets as widgets
#from IPython.display import display, clear_output
import cv2
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset
from engine import train_one_epoch, evaluate_hoi
from models import build_model
torch.set_grad_enabled(False);
# COCO classes
def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--lr_drop', default=60, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='dilation_tiny', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers_hopd', default=3, type=int,
                        help="Number of hopd decoding layers in the transformer")
    parser.add_argument('--dec_layers_interaction', default=3, type=int,
                        help="Number of interaction decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=64, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_feature_levels', default=3, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # HOI
    parser.add_argument('--num_obj_classes', type=int, default=80,
                        help="Number of object classes")
    parser.add_argument('--num_verb_classes', type=int, default=117,
                        help="Number of verb classes")
    parser.add_argument('--backbone_pretrained', type=str, default='',
                        help='Pretrained model path')
    parser.add_argument('--pretrained', type=str, default='',
                        help='Pretrained model path')
    parser.add_argument('--subject_category_id', default=0, type=int)
    parser.add_argument('--verb_loss_type', type=str, default='focal',
                        help='Loss type for the verb classification')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--use_matching', action='store_true',
                        help="Use obj/sub matching 2class loss in first decoder, default not use")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=2.5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=1, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_obj_class', default=1, type=float,
                        help="Object class coefficient in the matching cost")
    parser.add_argument('--set_cost_verb_class', default=1, type=float,
                        help="Verb class coefficient in the matching cost")
    parser.add_argument('--set_cost_matching', default=1, type=float,
                        help="Sub and obj box matching coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=2.5, type=float)
    parser.add_argument('--giou_loss_coef', default=1, type=float)
    parser.add_argument('--obj_loss_coef', default=1, type=float)
    parser.add_argument('--verb_loss_coef', default=2, type=float)
    parser.add_argument('--alpha', default=0.5, type=float, help='focal loss alpha')
    parser.add_argument('--matching_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='vcoco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--hoi_path', type=str, default='C:/Users/13220/Desktop/train')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='param/checkpoint_best.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # decoupling training parameters
    parser.add_argument('--freeze_mode', default=0, type=int)
    parser.add_argument('--obj_reweight', action='store_true')
    parser.add_argument('--verb_reweight', action='store_true')
    parser.add_argument('--use_static_weights', action='store_true',
                        help='use static weights or dynamic weights, default use dynamic')
    parser.add_argument('--queue_size', default=4704*1.0, type=float,
                        help='Maxsize of queue for obj and verb reweighting, default 1 epoch')
    parser.add_argument('--p_obj', default=0.7, type=float,
                        help='Reweighting parameter for obj')
    parser.add_argument('--p_verb', default=0.7, type=float,
                        help='Reweighting parameter for verb')

    # hoi eval parameters
    parser.add_argument('--use_nms_filter', action='store_true', help='Use pair nms filter, default not use')
    parser.add_argument('--thres_nms', default=0.5, type=float)
    parser.add_argument('--nms_alpha', default=1.0, type=float)
    parser.add_argument('--use_nms', action='store_true')
    parser.add_argument('--nms_beta', default=0.5, type=float)
    parser.add_argument('--json_file', default='results.json', type=str)
    parser.add_argument('--nms_thresh', default=0.5, type=float)
    parser.add_argument('--eval_extra', action='store_true')
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    #config = get_config(args)
    CLASSES1 = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
       'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
       'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

    CLASSES = [
        'adjust', 'assemble', 'block', 'blow', 'board', 'break', 'brush_with',
        'buy', 'carry', 'catch', 'chase', 'check', 'clean',
        'control', 'cook', 'cut', 'cut_with', 'direct', 'drag', 'dribble',
        'drink_with', 'drive', 'dry', 'eat', 'eat_at', 'exit', 'feed', 'fill',
        'flip', 'flush', 'fly', 'greet', 'grind', 'groom', 'herd', 'hit',
       'hold', 'hop_on', 'hose', 'hug', 'hunt',
        'inspect', 'install', 'jump', 'kick', 'kiss', 'lasso',
       'launch', 'lick', 'lie_on', 'lift', 'light', 'load', 'lose', 'make',
        'milk', 'move', 'no_interaction', 'open', 'operate', 'pack', 'paint',
        'park', 'pay', 'peel', 'pet', 'pick', 'pick_up',
        'point', 'pour', 'pull', 'push', 'race', 'read', 'release ', 'repair',
        'ride', 'row', 'run', 'sail', 'scratch', 'serve', 'set',
        'shear', 'sign', 'sip', 'sit_at', 'sit_on', 'slide',
        'smell', 'spin', 'squeeze', 'stab', 'stand_on', 'stand_under', 'stick',
       'stir', 'stop_at', 'straddle', 'swing', 'tag', 'talk_on', 'teach', 'text_on',
        'throw', 'tie', 'toast', 'train', 'turn', 'type_on',
        'walk', 'wash', 'watch', 'wave', 'wear', 'wield',
        'zip '
    ]

    # colors for visualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    print('create model')
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)

    # device = torch.cuda()
    model.eval()
    print('create image')
    # url = '/home/msl/hico_20160224_det/hico_20160224_det/images/test/HICO_test2015_00000555.jpg'
    dataset = ImageFolder('D:/dilation_CDN/dilation_CDN/train', transform=make_hico_transforms('val'))
    imgList = []
    name = []
    for i in range(len(dataset.imgs)):
        imgList.append(dataset.imgs[i][0])
        name.append(dataset.imgs[i][0].split('/')[-1])
    # imgList = np.array(imgList)
    for x in range(len(imgList)):
        im = Image.open(imgList[x])
        # original_image = cv2.imread(url)
        original_image = np.asarray(im)
        if original_image.shape[-1] == 3:
            # mean-std normalize the input image (batch-size: 1)
            img = transform(im).unsqueeze(0).to(device)
            # img = torch.from_numpy(img).to(device)
            # propagate through the model
            outputs = model(img)             #在这里存
            # keep only predictions with 0.7+ confidence
#           probas = outputs['pred_obj_logits'].softmax(-1)[0, :, :-1]
#            verbs = outputs['pred_verb_logits'].softmax(-1)[0, :, :-1]
#            keep = probas.max(-1).values > 0.7
#            keep2 = verbs.max(-1).values > 0.3


            # convert boxes from [0; 1] to image scales
#            bboxes_scaled_object = rescale_bboxes(outputs['pred_obj_boxes'][0, keep], im.size)
#            bboxes_scaled_human = rescale_bboxes(outputs['pred_sub_boxes'][0, keep], im.size)
            # 可视化检测结果
#            a = probas[keep]
#            b = verbs[keep2]
#            plot_results(im, b, bboxes_scaled_object)
            #plot_results(im, probas[keep], bboxes_scaled_object)


            # Detection - Visualize encoder-decoder multi-head attention weights

            #Here we visualize attention weights of the last decoder layer. This corresponds to visualizing,
            #for each detected objects, which part of the image the model was looking at to predict this specific bounding box and class.
            #conv_features, enc_attn_weights, dec_attn_weights, interaction_dec_attn_weights = [], [], [], []

            #hooks = [
            #    model.backbone[-2].register_forward_hook(
            #        lambda self, input, output: conv_features.append(output)
            #    ),
            #    model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            #        lambda self, input, output: enc_attn_weights.append(output[1])
            #    ),
            #    model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
            #        lambda self, input, output: dec_attn_weights.append(output[1])
            #    ),
            #    model.transformer.interaction_decoder.layers[-1].multihead_attn.register_forward_hook(
            #        lambda self, input, output: interaction_dec_attn_weights.append(output[1])
            #    ),
            #]

            # propagate through the model
            #outputs = model(img)

            #for hook in hooks:
             #   hook.remove()

            # don't need the list anymore
            #conv_features = conv_features[0]
            #enc_attn_weights = enc_attn_weights[0]
            #dec_attn_weights = dec_attn_weights[0]
            #interaction_dec_attn_weights = interaction_dec_attn_weights[0]
            #h, w = conv_features['layer3'].tensors.shape[-2:]

            # fig, axs = plt.subplots(ncols=len(bboxes_scaled_object), nrows=3, figsize=(22, 7))
            #colors = COLORS * 100
            #res=[]
            #Xmin=[]
            #Ymin=[]
            #Xmax=[]
            #Ymax=[]
            #for idx, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), bboxes_scaled_object):
            #    Xmin.append(xmin)
            #    Ymin.append(ymin)
            #    Xmax.append(xmax)
            #    Ymax.append(ymax)

                # ax = ax_i[0]
                # # ax.imshow(dec_attn_weights[0, idx].cpu().view(h, w))
                # ax.axis('off')
                # ax.set_title(f'query id: {idx.item()}')
                #
                # ax = ax_i[1]
            #    mask = norm_image(dec_attn_weights[0, idx].cpu().view(h, w).numpy())
            #    mask = cv2.resize(mask, (im.size[0], im.size[1]))
            #    res.append(mask)
            #    im_color = cv2.applyColorMap(np.uint8(mask), cv2.COLORMAP_JET)    # COLORMAP_OCEAN
            #    im_color = np.float32(im_color) / 255
            #    im_color = im_color[..., ::-1]  # gbr to rgb
                # ax.imshow(im_color)
                # ax.set_title('heatmap')

                # ax = ax_i[2]
            #    cam = im_color*255 + np.float32(original_image[..., ::-1])
                # ax.imshow(norm_image(cam))
                # ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                #                            fill=False, color='blue', linewidth=1))
                # ax.axis('off')
                # ax.set_title(CLASSES[probas[idx].argmax()])

            # fig.tight_layout()

            # 合并
          #  res_im_color = np.zeros((im.size[1],im.size[0]), dtype=float, order='C')
          #  for i in range(len(res)):
          #      res_im_color = res_im_color + res[i]
          #  im_color = cv2.applyColorMap(np.uint8(res_im_color), cv2.COLORMAP_JET)
          #  im_color = np.float32(im_color) / 255
          #  im_color = im_color[..., ::-1]  # gbr to rgb
          #  cam = (im_color * 255  + np.float32(original_image[..., ::-1]))[..., ::-1]
          #  cam = norm_image(cam)

            # plt.figure(figsize=(16, 10))
            # plt.imshow(cam)
            # ax = plt.gca()
           # colors = COLORS * 100
            #for p, (xmin, ymin, xmax, ymax), c in zip(probas[keep], bboxes_scaled_object.tolist(), colors):
                # ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                #                            fill=False, color=c, linewidth=3))
            #    cam = cv2.rectangle(cam, (int(xmin), int(ymin)), (int(xmax), int(ymax)),  (0, 0, 255))
                # cl = p.argmax()

           # for p, (xmin, ymin, xmax, ymax), c in zip(probas[keep], bboxes_scaled_human.tolist(), colors):
                 # ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                 #                           fill=False, color=c, linewidth=3))
            #     cam = cv2.rectangle(cam, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0))
                 # cl = p.argmax()
                # text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
                # cam = cv2.putText(cam, text, (int(xmin),int(ymin)+15),cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
                # ax.text(xmin, ymin, text, fontsize=15,
                #         bbox=dict(facecolor='yellow', alpha=0.5))
            # plt.axis('off')
            # plt.show()
            #cv2.imwrite('./result/'+name[x], cam)

