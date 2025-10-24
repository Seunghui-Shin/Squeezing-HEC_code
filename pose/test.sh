#!/bin/sh
i=1

while [ $i -lt 100 ]

do

        python3 /home/robot_ws/src/code/pose/launch_ros_discrete_sac.py --num $i
        
        i=$(($i+1))

done