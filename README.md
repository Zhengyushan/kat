## KAT: kernel attention Transformer for histopathology whole slide image classification

This is a PyTorch implementation of the paper [KAT]( ):

### Data preparation
1. Data structure
The structure of the whole slide image dataset to run the code.
We take a lung cancer datset collected from TCGA as the example.
```
./data                                                              # The directory of the data.
├─ TCGA-55-8510-01Z-00-DX1.BB1EAC72-6215-400B-BCBF-E3D51A60182D     # The directory for a slide.
│  ├─ Large                                                         # The directory of image tiles in Level 0 (40X lens).
│  │  ├─ 0000_0000.jpg                                              # The image tile in Row 0 and Column 0.
│  │  ├─ 0000_0001.jpg                                              # The image tile in Row 0 and Column 1.
│  │  └─ ...
│  ├─ Medium                                                        # The directory of image tiles in Level 1 (20X lens).
│  │  ├─ 0000_0000.jpg
│  │  ├─ 0000_0001.jpg
│  │  └─ ...
│  ├─ Small                                                         # The directory of image tiles in Level 2 (10X lens).
│  │  ├─ 0000_0000.jpg
│  │  ├─ 0000_0001.jpg
│  │  └─ ...
│  ├─ Overview                                                      # The directory of image tiles in Level 3 (5X lens).
│  │  ├─ 0000_0000.jpg
│  │  ├─ 0000_0001.jpg
│  │  └─ ...
│  └─ Overview.jpg                                                  # The thumbnail of the WSI in Level 3.     
│     
├─ TCGA-44-3919-01A-01-BS1.9251d6ad-dab8-42fd-836d-1b18e5d2afed
└─ ...
```
2. Generate configuration file for the dataset
Run dataset/configure_dataset.py to configure the dataset.

### Train
Run the codes on a single GPU:
```
CONFIG_FILE='configs/tcga_lung.yaml'
WOKERS=8
GPU=0

python cnn_sample.py --cfg $CONFIG_FILE --num-workers $WOKERS
for((FOLD=0;FOLD<5;FOLD++)); 
do
    python cnn_train_cl.py --cfg $CONFIG_FILE --fold $FOLD\
        --epochs 21 --batch-size 100 --num-workers $WOKERS --weighted-sample\
        --fix-pred-lr --eval-freq 2 --gpu $GPU

    python cnn_wsi_encode.py --cfg $CONFIG_FILE --fold $FOLD\
        --batch-size 512 --num-workers $WOKERS --gpu $GPU

    python kat_train.py --cfg $CONFIG_FILE --fold $FOLD\
        --num-epochs 101 --batch-size 32 --num-workers $WOKERS  --weighted-sample\
        --eval-freq 5 --gpu $GPU
done 

```

Run the codes multiple GPUs:
```
CONFIG_FILE='configs/tcga_lung.yaml'
WOKERS=8
WORLD_SIZE=1

python cnn_sample.py --cfg $CONFIG_FILE --num-workers $WOKERS

for((FOLD=0;FOLD<5;FOLD++)); 
do
    python cnn_train_cl.py --cfg $CONFIG_FILE --fold $FOLD\
        --epochs 21 --batch-size 400 --num-workers $WOKERS --weighted-sample\
        --fix-pred-lr --eval-freq 2\
        --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size $WORLD_SIZE --rank 0

    python cnn_wsi_encode.py --cfg $CONFIG_FILE --fold $FOLD\
        --batch-size 512 --num-workers $WOKERS\
        --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size $WORLD_SIZE --rank 0

     python kat_train.py --cfg $CONFIG_FILE --fold $FOLD\
        --num-epochs 101 --batch-size 128 --num-workers $WOKERS  --weighted-sample --eval-freq 5\
        --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size $WORLD_SIZE --rank 0
done

```

If the code is helpful to your research, please cite:
```
@inproceedings{zheng2022kernel,
    author    = {Yushan Zheng, Jun Li, Jun Shi, Fengying Xie, Zhiguo Jiang},
    title     = {Kernel Attention Transformer (KAT) for Histopathology Whole Slide Image Classification},
    booktitle = {Medical Image Computing and Computer Assisted Intervention 
                -- MICCAI 2022},
    year      = {2022}
}
```