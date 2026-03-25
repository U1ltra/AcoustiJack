## Claim 1 - Attack Effectiveness

**Claimed Items**
1. Our attack achieves success across different types of tracking algorithms. 
2. Robust tracking designs does not withstand our attack.

Specific results to compare to in the manuscript
- Figure 10. *HighEndDrone*. *DaSiamRPN* and *UCMCTrack*. 
---

**Evaluation Overhead**

To limit evaluation overhead, we prepared the selectively scaled-down experiment as the following
1. A outdoor (*field*) pedestrian scenario only, which is the representitive application scenario for UAV tracking. Real-world uncertainties including motion uncertainties (e.g. target speed and direction, flight disturbances), object appearance variations, environmental changes are still varied across the trials. 
2. Only *DaSiamRPN* (appearance-aware) and *UCMCTrack* (motion-based) algorithms  to show that the attack succeeds regardless of (1) tracking algorithms types and (2) robustness-oriented design (e.g. distractor-aware or motion-compensation).

The scaled-down experiment should take about *~3h* to complete with an NVIDIA GPU. GPU is not required but it will be slower without it.

---
**Launch Evaluation**
1. Run `bash bash/claim1.sh`.
2. Run `python utils/eval/offline_eval.py --claim 1`
3. Inspect the printed results from the terminal
