python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --use_env \
        main.py \
        --pretrained params/dilation_tiny_coco_after_hico.pth \
        --output_dir logs/tiny_coco_after_hico \
        --dataset_file vcoco \
        --hoi_path /userhome/coco2014/ \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --backbone dilation_tiny \
        --num_queries 64 \
        --dec_layers_hopd 3 \
        --dec_layers_interaction 3 \
        --epochs 150 \
        --lr_drop 120 \
        --use_nms_filter