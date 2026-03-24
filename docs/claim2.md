## Claim 2: Attack Generalizability

**Claimed Items**
1. Our attack achieves success across UAV gimbal systems with different acoustic vulnerabilities. 

Specific results to compare to in the manuscript
- Figure 10. *MidEndDrone*. *DaSiamRPN* and *UCMCTrack*. 
---

**Evaluation Overhead**

To limit evaluation overhead, we prepared the selectively scaled-down experiment as the following
1. The pedestrian scenario only, which is the representitive application scenario for UAV tracking.
2. One outdoor environment (*field*) and one indoor environment (*warehouse*), to account for necessary real-world uncertainties. 
3. Only *DaSiamRPN* (appearance-aware) and *UCMCTrack* (motion-based) algorithms to account for different tracking algorithm types.

The scaled-down experiment should take about *~5h* to complete with an NVIDIA GPU. GPU is not required but it will be slower without it.

---
**Launch Evaluation**
1. Run `bash bash/claim2.sh`.
2. Run `python utils/eval/offline_eval.py --claim 2`
3. Inspect the printed results from the terminal

---
**Known Issues**
- The code may crush. As far as our knowledge is concern, we think the problem is in the specific GPU and its supporting software we are using instead of our implementation. We are not sure if this problem may persist on another machine. So we recommend making the /root/exp in the docker container a shared directory with the host machine to avoid losing the logs when crushes.
- Our project involves complex interaction between Gazebo physical simulator, PX4-Autopilot flight control stack, ROS2, and python scripts that connects them. Therefore, running the project on another machine might introduce certain unexpected behaviors. For example, if the physical simulator is slowed down significantly due to limited computation, it might introduce delay into interaction with other components, which might lead to difference in results. While we don't expect it to be significant given the limited resources the project require, it is worthy noting. 