# Squeezing the Last Drop of Accuracy: Hand-Eye Calibration via Deep Reinforcement Learning-Guided Pose Tuning 

## 1. Usage

Our code is tested on Ubuntu 20.04 and ROS Noetic.

### 1.1 Build Instructions

- Conda environment
```
conda env create -f hec.yaml
```

### 1.2 Simulation

Set up your desired simulation configuration (e.g., robot, camera, etc.).


### 1.3 Pose Network and DRL

If you want to use DenseFusion for pose network, yo should follow [DenseFusion](https://github.com/j96w/DenseFusion).

Also, you sholud follow [Discrete SAC](https://github.com/BY571/SAC_discrete) and [Learn-to-Calibrate](https://github.com/ethz-asl/Learn-to-Calibrate/tree/master?tab=readme-ov-file) for DRL.

- After Installation, Clone the repository and catkin build:
```
mkdir robot_ws
cd robot_ws
mkdir src
cd src
git clone https://github.com/Seunghui-Shin/Squeezing-HEC_code.git
cd Squeezing-HEC_code
cd ../..
catkin build
```

- Kinova Gen3 Lite end-effector weights using Densefusion are [here](https://drive.google.com/drive/folders/1iTDNV9EuPDNXYYQFyqe7nQip-gd8v-8i?usp=drive_link).
- You should move the weights to pose/weights.


## 2. Running the code

### 2.1 Gazebo simulation

Run your simulation environment.
```
ex) roslaunch kortex_gazebo spawn_kortex_robot_realsense.launch arm:=gen3_lite
```
   
### 2.2.1 Training Policy
```
conda activate hec
cd robot_ws/src/Squeezing-HEC_code/hec
python3 python/train_policies/RL_algo_kinova_discrete_sac.py
```
### 2.2.2 Testing Policy
```
conda activate hec
cd robot_ws/src/Squeezing-HEC_code/hec
python3 python/test_policies/RL_algo_test_kinova_discrete_sac.py
```

### 2.3 Pose Network and Robot Control
```
conda activate hec
cd robot_ws/src/Squeezing-HEC_code
./pose/train.sh
```
We use Kinova Gen3 Lite manipulator and DenseFusion for posenet.

So, refer to the code to integrate your robot with the desired posenet.

## 3. License

License under the [MIT Ricense](https://github.com/Seunghui-Shin/Squeezing-HEC_code/blob/main/license/MIT/LICENSE.txt)

License under the [Kinova inc](https://github.com/Seunghui-Shin/Squeezing-HEC_code/blob/main/license/kinova/LICENSE.txt)


## 4. Code reference:

Our code is based on the following repositories:

- [ros_kortex](https://github.com/Kinovarobotics/ros_kortex)
- [realsense](https://github.com/issaiass/realsense2_description)
- [realsense sdk](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)
- [ArUco marker](https://github.com/ValerioMagnago/aruco_description)
- [DenseFusion](https://github.com/j96w/DenseFusion)
- [Discrete SAC](https://github.com/BY571/SAC_discrete)
- [Learn-to-Calibrate](https://github.com/ethz-asl/Learn-to-Calibrate/tree/master?tab=readme-ov-file)
