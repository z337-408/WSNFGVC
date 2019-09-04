# Web-Supervised Network for Fine-Grained Visual Classification

Introduction
------------
This is the source code for our paper **Web-Supervised Network for Fine-Grained Visual Classification**

Network Architecture
--------------------
The architecture of our proposed peer-learning model is as follows
![network](network.png)

Installation
------------
After creating a virtual environment of python 3.7, run `pip install -r requirements.txt` to install all dependencies

How to use
------------
The code is currently tested only on GPU
* **Data Preparation**
    - Download data into project root directory and uncompress them using
        ```
        wget https://wsnfg.oss-cn-hongkong.aliyuncs.com/web-bird.tar.gz
        wget https://wsnfg.oss-cn-hongkong.aliyuncs.com/web-car.tar.gz
        wget https://wsnfg.oss-cn-hongkong.aliyuncs.com/web-aircraft.tar.gz
        tar -xvf web-bird.tar.gz
        tar -xvf web-car.tar.gz
        tar -xvf aircraft-car.tar.gz
        ```
* **Model Preparation**
    - Download the following trained models into `model/` using  
        - Bird:      ``` wget https://wsnfg.oss-cn-hongkong.aliyuncs.com/bird-77.22.pth     ```
* **Demo**

    - If you just want to do a quick test on the model and check the final fine-grained recognition performance, please follow subsequent steps
  
      - Create a soft link for data by `ln -s web-bird bird`
      - Activate virtual environment (e.g. conda)
      - Modify `CUDA_VISIBLE_DEVICES` to proper cuda device id in `cub200_demo.sh` 
      - Run demo using `bash cub200_demo.sh`

* **Source Code**

    - If you want to train the whole network from begining using source code on the web fine-grained dataset, please follow subsequent steps
    
      - Create soft link to dataset by
       ```
             ln -s web-bird bird
             ln -s web-car car
             ln -s web-aircraft aircraft
       ```

      - Modify `CUDA_VISIBLE_DEVICES` to proper cuda device id in 
          ```
          cub200_run.sh
          car196_run.sh
          aircraft100_run.sh
          ```
      - Activate virtual environment(e.g. conda) and then run the script
       ```
          bash cub200_train.sh
          bash car196_run.sh
          bash aircraft100_run.sh
       ```
