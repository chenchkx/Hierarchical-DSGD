
# AlexNet
python main_amp.py --dataset_name "CIFAR10" --image_size 64 --batch_size 64 --mode "ring" --size 16 --lr 0.1 --model "AlexNet_M" --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
python main_amp.py --dataset_name "CIFAR10" --image_size 64 --batch_size 512 --mode "ring" --size 16 --lr 0.8 --model "AlexNet_M" --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0

python main_amp.py --dataset_name "CIFAR10" --image_size 64 --batch_size 64 --mode "ring" --size 16 --lr 0.2 --model "AlexNet_M" --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
python main_amp.py --dataset_name "CIFAR10" --image_size 64 --batch_size 512 --mode "ring" --size 16 --lr 1.6 --model "AlexNet_M" --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0

# with warmup
python main_amp.py --dataset_name "CIFAR10" --image_size 64 --batch_size 64 --mode "ring" --size 16 --lr 0.1 --model "AlexNet_M" --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
python main_amp.py --dataset_name "CIFAR10" --image_size 64 --batch_size 512 --mode "ring" --size 16 --lr 0.8 --model "AlexNet_M" --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0

python main_amp.py --dataset_name "CIFAR10" --image_size 64 --batch_size 64 --mode "ring" --size 16 --lr 0.2 --model "AlexNet_M" --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
python main_amp.py --dataset_name "CIFAR10" --image_size 64 --batch_size 512 --mode "ring" --size 16 --lr 1.6 --model "AlexNet_M" --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0


# ResNet18_M
python main_amp.py --dataset_name "CIFAR10" --image_size 56 --batch_size 64 --mode "ring" --size 16  --lr 0.1 --model "ResNet18_M" --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
python main_amp.py --dataset_name "CIFAR10" --image_size 56 --batch_size 512 --mode "ring" --size 16  --lr 0.8 --model "ResNet18_M" --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0

# with warmup
python main_amp.py --dataset_name "CIFAR10" --image_size 56 --batch_size 64 --mode "ring" --size 16 --lr 0.1 --model "ResNet18_M" --warmup_step 300 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
python main_amp.py --dataset_name "CIFAR10" --image_size 56 --batch_size 512 --mode "ring" --size 16 --lr 0.8 --model "ResNet18_M" --warmup_step 300 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0


# DenseNet121_M
python main_amp.py --dataset_name "CIFAR10" --image_size 56 --batch_size 64 --mode "ring" --size 16  --lr 0.1 --model "DenseNet121_M" --warmup_step 500 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
python main_amp.py --dataset_name "CIFAR10" --image_size 56 --batch_size 512 --mode "ring" --size 16  --lr 0.8 --model "DenseNet121_M" --warmup_step 500 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
