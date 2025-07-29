# ADeT-Net
## Pipeline

![framework](ADeT-Net/figs/1.jpg)


## Installation
1. Clone the repository.
    ```bash
   https://github.com/songshuaitian/API-Net
    ```

2. Install PyTorch 1.13.0 and torchvision 0.14.0.
    ```bash
    conda install -c pytorch pytorch torchvision
    ```

3. Install the other dependencies.
    ```bash
    pip install -r requirements.txt
    ```

## Prepare
- **RTTS** can be downloaded from [here.](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2)
- **RIDCP500** can be downloaded from [RIDCP's Repo](https://github.com/RQ-Wu/RIDCP_dehazing)
- **OurData** can be downloaded from (Link：https://pan.baidu.com/s/1llt0wjEO3STLjoRgXxp-NA?pwd=uhsr 
code：uhsr)
- **weight** can be downloaded from (Link：https://pan.baidu.com/s/1MsmIahkQbjGE659sFjLbzg?pwd=sfiu 
code：sfiu)
- The final file path should be the same as the following (please check it carefully):

```
┬─ save_models
│   ├─ rtts
│   │   ├─ API-Net.pth
│   │   └─ ... (model name)
│   └─ ... (exp name)
└─ data
    ├─ rgb500
    │   └─ ... (image filename)
    ├─ depth500
    │   └─ ... (image filename)
    ├─ Seggray
    │   └─ ... (image filename)
    ├─ RTTS
    │   └─ hazy
    │   │    └─ ... (image filename)
    │   └─ Seggray
    │       └─ ... (image filename)
    └─ ... (dataset name)
```

## Training

`configs`

After adjusting the settings, use the following script to initiate the training of the model:

```
CUDA_VISIBLE_DEVICES=X python train.py
```

For example：

```
CUDA_VISIBLE_DEVICES=0 python train.py
```

## Evaluation

Run the following script to evaluate the trained model with a single GPU.


```sh
CUDA_VISIBLE_DEVICES=X python test.py
```

For example：

```sh
CUDA_VISIBLE_DEVICES=0 python test.py
```


# Contact:
    Don't hesitate to contact me if you meet any problems when using this code.

    Song Shuaitian
    Faculty of Information Engineering and Automation
    Kunming University of Science and Technology                                                           
    Email: songshuaitiann@163.com
