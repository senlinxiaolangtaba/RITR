python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --use_env \
        main.py \
        --pretrained logs/tiny_hico/checkpoint_best.pth \
        --output_dir logs/tiny_hico \
        --dataset_file hico \
        --hoi_path /userhome/dataset/hico_20160224_det/ \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --start_epoch 0 \
        --backbone dilation_tiny \
        --num_queries 64 \
        --dec_layers_hopd 3 \
        --dec_layers_interaction 3 \
        --eval \
        --use_nms_filter
        