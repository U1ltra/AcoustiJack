## Claim 1 - Attack Effectiveness

**Claimed Items**
1. Our attack achieves success across different types of tracking algorithms.
2. Robust tracking designs do not withstand our attack.

Specific results to compare to in the manuscript:
- **Figure 10**, *HighEndDrone* column, rows *DaSiamRPN* and *UCMCTrack*.

Expected values (Hijack rate / Track loss rate):

| Tracker | Hijack rate | Track loss rate |
|---------|------------|-----------------|
| UCMCTrack | 78.5% | 98.1% |
| DaSiamRPN | 90.0% | 100% |
---

**Evaluation Overhead**

To limit evaluation overhead, we prepared the selectively scaled-down experiment as the following
1. Only *DaSiamRPN* (appearance-aware) and *UCMCTrack* (motion-based) algorithms  to show that the attack succeeds regardless of (1) tracking algorithms types and (2) robustness-oriented design (e.g. distractor-aware or motion-compensation).
2. For *DaSiamRPN* (appearance-aware), we select an outdoor (*field*) pedestrian scenario only, which is the representitive application scenario for UAV tracking. Real-world uncertainties including motion uncertainties (e.g. target speed and direction, flight disturbances), object appearance variations, environmental changes are still varied across the trials. 
3. For *UCMCTrack* (motion-based), we select an outdoor (*field*) pedestrian and car (*raceway*) scenarios to account for the diverse motion model of pedestrian and car.

The scaled-down experiment should take about *~4h* to complete with an NVIDIA GPU. GPU is not required but it will be slower without it.

---
**Launch Evaluation**
1. Run `bash bash/claim1.sh`.
2. Run `python utils/eval/offline_eval.py --claim 1`
3. Inspect the printed summary at the end of the output, e.g.:
   ```
   Tracker performance summary:
   dasiam: Hijack rate: 0.90, Track loss rate: 0.10
   ucmc: Hijack rate: 0.78, Track loss rate: 0.22
   ```
   Compare the **Hijack rate** values to Figure 10 (*HighEndDrone* column) in the paper.

> **Note**: Trials that fail to load (e.g., `Failed to load simulation state`) correspond to trials that did not initialize correctly due to integration instability — they are skipped automatically and do not affect valid results. See [Known Issues](../README.md#known-issues) for how to rerun individual failed trials.
