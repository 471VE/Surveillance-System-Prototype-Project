# Surveillance System Prototype Project

This is surveillance system prototype based on DeepSORT that has support for better detection and ReID models than the original repository. It uses videos from MOT challenge to demonstrate efficiency and for benchmarking. There are several ways to run this project.

## Method 1 (local)

First, clone the repository to your local machine:

```
git clone https://github.com/471VE/Surveillance-System-Prototype-Project.git
```

Second, install necessary requirements (use ```pip3``` on Linux):
```
pip install -r requirements.txt

pip install -r requirements_rep.txt
```

Third, download and extarct model weights and MOT videos:
```
gdown 17I6a6P7mitj2scm4MuCCdHyfvhhBqFTB

gdown 1yKP1alYep2pO0w7woGW48YFvt7dquhzM

unzip weights_surveillance.zip

unzip MOT_custom.zip
```

Now, you can run the following scripts to process videos and evaluating HOTA:

```
python real_time_single.py --sequence_dir=./MOT_custom/KITTI-17

python real_time_all.py --mot_dir=./MOT_custom

python HOTA_evaluation/run_mot_challenge.py
```

<b>WARNING:</b> in order to evaluate HOTA on the videos, all of the videos in ```MOT_custom``` directory must be processed for every tracker that is to be evaluated.

You may also use ```demo.ipynb``` Jupyter Notebook for the same perposes.

## Method 2 (on Google Colab)

Alternatively, you can open [```demo_colab.ipynb```](https://github.com/471VE/Surveillance-System-Prototype-Project/blob/master/demo_colab.ipynb) notebook directly on github and follow the instructions there.