```shell
CUDA_VISIBLE_DEVICES=0,1 python main.py \
--model_name=bert-base-chinese \
--tokenizer_name=bert-base-chinese \
--project_name=gaiic2022-visualbert-base \
--num_workers=4 \
--gpus=2 \
--multi_gpu_strategy=dp \
--batch_size=128 \
--wandb \
--lr=0.0001 \
--ft_lr=0.0001 \
--dropout=0.1 \
--dataset_version=2 \
--max_epochs=30 \
--min_epochs=10 \
--image_token_size=1 \
--focal_loss \
--scheduler=cosine \
--use_pre_trained \
--train_file="/workspace/com/jd/xab/data/train_coarse.txt,/workspace/com/jd/xab/data/train_fine.txt.00" \
--dev_file="/workspace/com/jd/xab/data/train_fine.txt.01"
```
