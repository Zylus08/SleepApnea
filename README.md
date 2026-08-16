# A Comprehensive Multimodal Dataset for Investigating Cognitive Impairment in Obstructive Sleep Apnea
﻿
## 1. Overview
The dataset includes behavioral, polysomnography (PSG), neuroimaging, and questionnaire recordings collected from 142 participants, including patients with OSA and matched healthy controls.All data have been rigorously curated to comply with the Brain Imaging Data Structure (BIDS) standard.
﻿
## 2. Experimental Design (Longitudinal Sessions)
The experimental protocol utilizes a test-retest multi-session design spanning a 24-hour period to support the investigation of cognitive impairment in OSA.The directory structure is organized into the following sessions (`ses-`):
* `ses-preSleep`: Evening baseline cognitive assessment.
* `ses-nightSleep`: Nocturnal polysomnography (PSG) recording.
* `ses-postSleep`: Morning cognitive assessment immediately following nocturnal sleep.
* `ses-preNap`: Early afternoon cognitive assessment prior to a diurnal nap.
* `ses-postNap`: Late afternoon assessment following the nap, including structural and resting-state functional MRI scans.
﻿
## 3. Cognitive Assessment Battery (`task-cognitive`)
During the waking sessions (`ses-preSleep`, `ses-postSleep`, `ses-preNap`, `ses-postNap`), participants completed a continuous cognitive assessment battery:
1.  **PPT**: Picture Pairing Task
2.  **SART**: Sustained Attention to Response Task
3.  **MST**: Motor Skills Task
4.  **PVT**: Psychomotor Vigilance Task
﻿
## 4. Data Modalities & Technical Notes
* **Polysomnography (PSG)**: Standard multi-channel PSG (`.edf`) was recorded continuously during the `ses-nightSleep` session.
* **Magnetic Resonance Imaging (MRI)**: High-resolution T1-weighted structural scans (`anat`) and resting-state BOLD functional scans (`func`) were acquired once at the `ses-postNap` timepoint to investigate structural divergence and cognitive network reorganization.
* **Behavioral Data (`beh`)**: Task performance metrics (e.g., reaction times, error rates) are provided in tab-separated `.tsv` format, isolated by session.
﻿
## 5. Directory Structure Highlight
```text
dataset/
├── README.md
├── dataset_description.json
├── sub-001/
│   ├── ses-preSleep/  (Behavioral)
│   ├── ses-nightSleep/ (PSG EDF)
│   ├── ses-postSleep/ (Behavioral)
│   ├── ses-preNap/    (Behavioral)
│   └── ses-postNap/   (Behavioral, T1w MRI, rs-fMRI)
└── ...
```
﻿
## 6. Contributors
* Wei Guo, , Wei Tian, Wanqi Chen, Tao Jiang, Dong Chen, Fengyu Cong, and Fan Li
﻿
**Institution:** School of Biomedical Engineering, Faculty of Medicine, Dalian University of Technology
﻿
## 7. License and Usage
This dataset is released under the CC0 license. If you use this data in your research or utilize our custom analysis pipelines, please cite the accompanying dataset descriptor publication and the provided Dataset DOI.
﻿