# Temperature Sweep Results

Sweep directory: `outputs/config_sweeps/20260521_211942`

Stage: `distillation`

Completed runs at commit time: 6 of 8

| Run | Temperature | Epoch 1 Val Loss | Epoch 2 Val Loss | Epoch 3 Val Loss | Final Perplexity | Checkpoint |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.75 | 3.4919 | 2.9461 | 2.6767 | 14.50 | `outputs/config_sweeps/20260521_211942/checkpoints/student_distilled_temp_0p75.pt` |
| 2 | 1.00 | 3.2657 | 2.7859 | 2.5247 | 12.46 | `outputs/config_sweeps/20260521_211942/checkpoints/student_distilled_temp_1p0.pt` |
| 3 | 1.25 | 3.3357 | 2.7876 | 2.5208 | 12.41 | `outputs/config_sweeps/20260521_211942/checkpoints/student_distilled_temp_1p25.pt` |
| 4 | 1.50 | 3.4870 | 2.9188 | 2.6358 | 13.93 | `outputs/config_sweeps/20260521_211942/checkpoints/student_distilled_temp_1p5.pt` |
| 5 | 1.75 | 3.6938 | 3.0244 | 2.7217 | 15.18 | `outputs/config_sweeps/20260521_211942/checkpoints/student_distilled_temp_1p75.pt` |
| 6 | 2.00 | 3.8618 | 3.1546 | 2.8186 | 16.73 | `outputs/config_sweeps/20260521_211942/checkpoints/student_distilled_temp_2p0.pt` |

Best completed run so far: `temperature=1.25` with final perplexity `12.41`.

Intermediate takeaway: this sweep favors lower distillation temperatures. Results degrade noticeably from `temperature=1.5` onward, with `temperature=1.0` and `temperature=1.25` clearly outperforming the higher-temperature runs completed so far.
