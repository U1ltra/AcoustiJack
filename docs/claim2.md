## Claim 2: Attack Generalizability

**Claimed Items**
1. Our attack achieves success across UAV gimbal systems with different acoustic vulnerabilities. 

Specific results to compare to in the manuscript
- Figure 10. *MidEndDrone*. *DaSiamRPN* and *UCMCTrack*. 
---

**Evaluation Overhead**

To limit evaluation overhead, we prepared the selectively scaled-down experiment as the following
1. A outdoor (*field*) pedestrian scenario only, which is the representitive application scenario for UAV tracking. Real-world uncertainties including motion uncertainties (e.g. target speed and direction, flight disturbances), object appearance variations, environmental changes are still varied across the trials. 
2. Only *DaSiamRPN* (appearance-aware) and *UCMCTrack* (motion-based) algorithms to account for different tracking algorithm types.

The scaled-down experiment should take about *~2.5h* to complete with an NVIDIA GPU. GPU is not required but it will be slower without it.

---
**Launch Evaluation**
1. Run `bash bash/claim2.sh`.
2. Run `python utils/eval/offline_eval.py --claim 2`
3. Inspect the printed results from the terminal
