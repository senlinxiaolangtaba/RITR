import torch
# import models.swin_transformer
import argparse
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--lr_drop', default=60, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='dilation_small', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers_hopd', default=6, type=int,
                        help="Number of hopd decoding layers in the transformer")
    parser.add_argument('--dec_layers_interaction', default=6, type=int,
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
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # HOI
    parser.add_argument('--num_obj_classes', type=int, default=80,
                        help="Number of object classes")
    parser.add_argument('--num_verb_classes', type=int, default=117,
                        help="Number of verb classes")
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
    parser.add_argument('--dataset_file', default='hico')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--hoi_path', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--num_feature_levels', default=3, type=int,
                        help="Number of query slots")

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
    parser.add_argument('--queue_size', default=4704 * 1.0, type=float,
                        help='Maxsize of queue for obj and verb reweighting, default 1 epoch')
    parser.add_argument('--p_obj', default=0.7, type=float,
                        help='Reweighting parameter for obj')
    parser.add_argument('--p_verb', default=0.7, type=float,
                        help='Reweighting parameter for verb')

    # hoi eval parameters
    parser.add_argument('--use_nms_filter', action='store_true', help='Use pair nms filter, default not use')
    parser.add_argument('--thres_nms', default=0.7, type=float)
    parser.add_argument('--nms_alpha', default=1.0, type=float)
    parser.add_argument('--nms_beta', default=0.5, type=float)
    parser.add_argument('--json_file', default='results.json', type=str)

    return parser


if __name__ == '__main__':
    pretrained_dict_swin_transformer_revise = {}
    pretrained_interaction_model = {}
    neck_dict = {}
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    pretrained_dict_swin_transformer = torch.load(
        "params/ckpt_epoch_299.pth") #['state_dict']
    for key, value in pretrained_dict_swin_transformer["model"].items():
        pretrained_dict_swin_transformer_revise['backbone.0.body.' + key] = value
        if key[0:4] == 'neck':
            pretrained_dict_swin_transformer_revise['backbone.1.' + key] = value
    pretrained_dict_DETR = torch.load("params/detr-r50-pre-2stage-q64.pth")
    DETR_keys = list(pretrained_dict_DETR['model'].keys())
    for key, value in pretrained_dict_DETR['model'].items():
#         print(key)
        if key[0:28] == 'transformer.decoder.layers.0' or key[0:28] == 'transformer.decoder.layers.1' or key[0:28] == 'transformer.decoder.layers.2' :
            pretrained_interaction_model[key[0:12] + 'interaction_' + key[12:]] = value
        if key[0:28] == 'transformer.decoder.layers.3' or key[0:28] == 'transformer.decoder.layers.4' or key[0:28] == 'transformer.decoder.layers.5' :
            pretrained_interaction_model[key[0:12] + 'interaction_' + key[12:]] = value
    pretrained_interaction_model['transformer.interaction_decoder.norm.weight'] = pretrained_dict_DETR['model'][
        "transformer.decoder.norm.weight"]
    pretrained_interaction_model["transformer.interaction_decoder.norm.bias"] = pretrained_dict_DETR['model'][
        "transformer.decoder.norm.bias"]
    model, criterion, postprocessors = build_model(args)
    model_dict = model.state_dict()
    trans_keys = list(pretrained_dict_swin_transformer.keys())
    model_keys = list(model_dict.keys())
#     for key in model_keys:
#         print(key)
    pretrained_dict_backbone = {k: v for k, v in pretrained_dict_swin_transformer_revise.items() if k in model_dict}
    pretrainen_DETR = {k: v for k, v in pretrained_dict_DETR['model'].items() if k in model_dict}
    pretrained_model_dict = dict(pretrained_dict_backbone, **pretrainen_DETR)
    pretrained_model_dict = dict(pretrained_model_dict, **pretrained_interaction_model)

    # pretrained_model_dict.pop('input_proj.weight')
    # original_parameter = torch.load("logs/CDN_transformer_detection_base.pth")
    pretrained_model_dict["verb_class_embed.weight"] = model_dict["verb_class_embed.weight"]
    pretrained_model_dict["verb_class_embed.bias"] = model_dict["verb_class_embed.bias"]
    pretrained_model_dict["input_proj.weight"] = model_dict["input_proj.weight"]
    pretrained_model_dict["input_proj.bias"] = model_dict["input_proj.bias"]
    pretrained_model_dict["backbone.0.body.norm1.weight"] = model_dict["backbone.0.body.norm1.weight"]
    pretrained_model_dict["backbone.0.body.norm1.bias"] = model_dict["backbone.0.body.norm1.bias"]
    pretrained_model_dict["backbone.0.body.norm2.weight"] = model_dict["backbone.0.body.norm2.weight"]
    pretrained_model_dict["backbone.0.body.norm2.bias"] = model_dict["backbone.0.body.norm2.bias"]
    pretrained_model_dict["backbone.0.body.norm3.weight"] = model_dict["backbone.0.body.norm3.weight"]
    pretrained_model_dict["backbone.0.body.norm3.bias"] = model_dict["backbone.0.body.norm3.bias"]
    # for key, value in pretrained_model_dict.items():
#         print(key)
    model.load_state_dict(pretrained_model_dict, True)
    torch.save(model.state_dict(),
               "params/dilation_small_hico.pth")
    print('load model complete')
 # pretrained_dict_swin_transformer_revise['backbone.1.lateral_convs.0.conv.weight'] = pretrained_dict_swin_transformer_revise[
    #     'backbone.1.lateral_convs.2.conv.weight']
    # pretrained_dict_swin_transformer_revise['backbone.1.lateral_convs.0.conv.bias'] = pretrained_dict_swin_transformer_revise[
    #     'backbone.1.lateral_convs.2.conv.bias']
    # pretrained_dict_swin_transformer_revise['backbone.1.lateral_convs.1.conv.weight'] = pretrained_dict_swin_transformer_revise[
    #     'backbone.1.lateral_convs.3.conv.weight']
    # pretrained_dict_swin_transformer_revise['backbone.1.lateral_convs.1.conv.bias'] = pretrained_dict_swin_transformer_revise[
    #     'backbone.1.lateral_convs.3.conv.bias']
    #
    # pretrained_dict_swin_transformer_revise['backbone.1.fpn_convs.0.conv.weight'] = pretrained_dict_swin_transformer_revise[
    #     'backbone.1.fpn_convs.2.conv.weight']
    # pretrained_dict_swin_transformer_revise['backbone.1.fpn_convs.0.conv.bias'] = pretrained_dict_swin_transformer_revise[
    #     'backbone.1.fpn_convs.2.conv.bias']
    # pretrained_dict_swin_transformer_revise['backbone.1.fpn_convs.1.conv.weight'] = pretrained_dict_swin_transformer_revise[
    #     'backbone.1.fpn_convs.3.conv.weight']
    # pretrained_dict_swin_transformer_revise['backbone.1.fpn_convs.1.conv.bias'] = pretrained_dict_swin_transformer_revise[
    #     'backbone.1.fpn_convs.3.conv.bias']
    #
    # pretrained_dict_swin_transformer_revise.pop('backbone.1.lateral_convs.2.conv.weight')
    # pretrained_dict_swin_transformer_revise.pop('backbone.1.lateral_convs.3.conv.weight')
    # pretrained_dict_swin_transformer_revise.pop('backbone.1.lateral_convs.2.conv.bias')
    # pretrained_dict_swin_transformer_revise.pop('backbone.1.lateral_convs.3.conv.bias')
    #
    # pretrained_dict_swin_transformer_revise.pop('backbone.1.fpn_convs.2.conv.weight')
    # pretrained_dict_swin_transformer_revise.pop('backbone.1.fpn_convs.3.conv.weight')
    # pretrained_dict_swin_transformer_revise.pop('backbone.1.fpn_convs.2.conv.bias')
    # pretrained_dict_swin_transformer_revise.pop('backbone.1.fpn_convs.3.conv.bias')