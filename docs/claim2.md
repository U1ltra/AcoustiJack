**Claim 2: The attack demonstrates generalizability with stable success across two commercial gimbal camera systems, each exhibiting distinct acoustic vulnerabilities as discussed in Section 5.1.**

The script should take about 8h to complete (4h for each 54 trials) with an NVIDIA GPU. GPU is not required but it will be slower without it.

These experiments are configured with acoustic vulnerability physically profiled from *MidEndDrone*. We selectively run experiments on DaSiamRPN (appearance-based) and UCMCTrack (motion-based), which are designed robustness against visual distractors and camera motions. This demonstrates the generalizability of our attack.

Running claim2.sh should generate another set of experiment logs under /root/exp. The configurations are already set to align with the *MidEndDrone* column in **Table 1**. To obtain the evaluation metric for each of the tracking algorithm, 
1) fill the *exp_names* list with the newly generated trace names in /root/exp
2) run util/offline_eval.py 
3) attack success rate for each of the printed to the terminal -> expected to be comparable to **Table 1** *MidEndDrone*

Known issues:
- The code may crush. As far as our knowledge is concern, we think the problem is in the specific GPU and its supporting software we are using instead of our implementation. We are not sure if this problem may persist on another machine. So we recommend making the /root/exp in the docker container a shared directory with the host machine to avoid losing the logs when crushes.
- Our project involves complex interaction between Gazebo physical simulator, PX4-Autopilot flight control stack, ROS2, and python scripts that connects them. Therefore, running the project on another machine might introduce certain unexpected behaviors. For example, if the physical simulator is slowed down significantly due to limited computation, it might introduce delay into interaction with other components, which might lead to difference in results. While we don't expect it to be significant given the limited resources the project require, it is worthy noting. 