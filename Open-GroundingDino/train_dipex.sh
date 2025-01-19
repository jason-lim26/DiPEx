python -u main.py \
        --amp \
        --output_dir logs_dipex \
        -c config/cfg_coco.py \
        --datasets config/datasets_caod.json \
        --pretrain_model_path /path/to/groundingdino_swint_ogc.pth \
        --options text_encoder_type=bert-base-uncased \