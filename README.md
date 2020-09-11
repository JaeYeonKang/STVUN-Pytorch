# STVUN: Deep Space-Time Video Upsampling Networks

This repository is for STVUN introduced in the following paper

Jaeyeon Kang, Younghyun Jo, Seoung Wug Oh, Peter Vajda, and Seon Joo Kim. "STVUN: Deep Space-Time Video Upsampling Networks", ECCV 2020.
[PDF](https://arxiv.org/abs/2004.024322), [Video](https://www.youtube.com/watch?v=ZQoGbN16zKk)



![Alt text](/imgs/teaser.PNG)

## Dependencies

    Python>=3.6.8, Pytorch>=1.3, CUDA version>= 10.2 


## Quickstart (Test models)

1. Clone this github repo. 

       git clone https://github.com/JaeYeonKang/STVUN-Pytorch
       cd STVUN-PYtorch

2. Compile the correlation package.

        cd networks/correlation_package
        python setup.py install
        
 3. Place your test dataset in './test' folder. (e.g. ./test/Vid4)
 
 4. Download our pretrained models from [link](https://drive.google.com). Then, place the models in ./pretrained_model
 
 5. Run demo. 
 
        python demo.py --data_dir $DATA_DIR$ \
        --save_dir $SAVE_DIR$ --pre_train $PRETRAINED_MODEL$ \
        --time_step $TIME_STEP$ 
        
      + DATA_DIR : path to test dataset
      + SAVE_DIR : path to save results
      + PRETRAINED_MODEL : path to pretrained model
      + TIME_STEP : the number of intermediate frames to generate
 
      For example,
 
        python demo.py --data_dir ./test/Vid4 \
        --save_dir ./results --pre_train ./pretrained_model/STVUN.pth \
        --time_step 3 
        
 

 ## Space-Time Video Testset(STVT) dataset
 
We collect Space-Time Video Test(STVT) dataset that consists of 12 dynamic scenes with both various motions
and spatial details for the joint upsampling evaluation. Each scene has at least 50 frames. 
You can download our STVT dataset from [link](https://drive.google.com/file/d/1nHuxP2t9HiqFjsbJdBPi7XbTdDzJAQoh/view?usp=sharing)

![Alt text](/imgs/STVT.PNG)


## Citation

If you use any part of this code in your research, please cite our paper

    @article{kang2020deep,
    title={Deep Space-Time Video Upsampling Networks},
    author={Kang, Jaeyeon and Jo, Younghyun and Oh, Seoung Wug and Vajda, Peter and Kim, Seon Joo},
    journal={arXiv preprint arXiv:2004.02432},
    year={2020}
    }

