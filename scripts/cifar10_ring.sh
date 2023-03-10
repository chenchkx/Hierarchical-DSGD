
# CIFAR10 training 50000
# 50000/(512*16) = 6.10
# 50000/(64*16)  = 48.8

## AlexNet
python main_hierarchical.py --dataset_name "CIFAR10" --image_size 64 --batch_size 64 --mode "ring" --size 16 --lr 0.1 --model "AlexNet_M" --warmup_step 0 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
python main_hierarchical.py --dataset_name "CIFAR10" --image_size 64 --batch_size 512 --mode "ring" --size 16 --lr 0.8 --model "AlexNet_M" --warmup_step 0 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0


## DenseNet121_M
# training with amp（automatic mixed precision）
python main.py --dataset_name "CIFAR10" --image_size 56 --batch_size 64 --mode "ring" --size 16  --lr 0.1 --model "DenseNet121_M" --warmup_step 60 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0 --amp
python main.py --dataset_name "CIFAR10" --image_size 56 --batch_size 512 --mode "ring" --size 16  --lr 0.8 --model "DenseNet121_M" --warmup_step 60 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0 --amp
