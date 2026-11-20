# Baselines

Each baseline has its own folder. The README in the folder describes how to install, prepare the data for and train that baseline.

| Family | Baseline | Folder |
|---|---|---|
| Standard IL | ACT | [ACT](policy/ACT) |
| Standard IL | Diffusion Policy | [DP](policy/DP) |
| Multi-agent IL | LatentToM | [LatentToM](policy/LatentToM) |
| Multi-agent IL | GauDP | [GauDP](policy/GauDP) |
| VLA | pi0.5 | [Pi_05](policy/Pi_05) |
| VLA | GR00T N1.7 | [GR00T_N17](policy/GR00T_N17) |
| VLA | Psi0 | [Psi0](policy/Psi0) |
| WAM | Fast-WAM | [FastWAM](policy/FastWAM) |

Every folder has the same three scripts.

| Script | Does |
|---|---|
| `install.sh` | builds the environment of the baseline and downloads its pretrained weights |
| `process_data.sh` | converts the demonstrations into the format the baseline trains on |
| `train.sh` | trains the baseline |
