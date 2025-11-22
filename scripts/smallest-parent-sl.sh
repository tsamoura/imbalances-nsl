###### SMALLEST-PARENT, IR = 0,5,15,50,70 SL loss  

export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=1 python -u train_smallest_parent.py --exp-dir experiment/MIPLL-SL-SMALLEST-PARENT-CIFAR-10-NS=10000-IR=5-seed2 \
--num-class 10 --imb_type exp --imb_ratio 5 --size_partial_dataset 3000 --batch-size 64 --save_ckpt --epochs 100 --seed 2

export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=1 python -u train_smallest_parent.py --exp-dir experiment/MMIPLL-SL-SMALLEST-PARENT-CIFAR-10-NS=10000-IR=15-seed2 \
--num-class 10 --imb_type exp --imb_ratio 15 --size_partial_dataset 3000 --batch-size 64 --save_ckpt --epochs 100 --seed 2


export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=1 python -u train_smallest_parent.py --exp-dir experiment/MIPLL-SL-SMALLEST-PARENT-CIFAR-10-NS=10000-IR=50-seed2 \
--num-class 10 --imb_type exp --imb_ratio 50 --size_partial_dataset 3000 --batch-size 64 --save_ckpt --epochs 100 --seed 2


export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=1 python -u train_smallest_parent.py --exp-dir experiment/MIPLL-SL-SMALLEST-PARENT-CIFAR-10-NS=10000-ORIGINAL-seed2 \
--num-class 10 --imb_type original --size_partial_dataset 3000 --batch-size 64 --save_ckpt --epochs 100 --seed 2
