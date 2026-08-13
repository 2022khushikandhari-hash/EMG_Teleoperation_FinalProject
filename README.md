````markdown
# BE Capstone Project

## Project Title

EMG based Teleoperation
Yt link :https://youtu.be/hXBAcoLtE1I?si=5lQOlRrPdAa18sjS
---

## Team Details

|Sr.No.|Name of Student.  |Roll No.| Branch | Email ID                         |
|------|------------------|--------|--------|----------------------------------|
| 1    |Khushi Kandhari   |  06    | AURO   | 2022.khushi.kandhar@ves.ac.in    |
| 2    |Anaga Bhat        |  39    | AURO   | 2023.anaga.bhat@ves.ac.in        |
| 3    |Tejasvini Bachhav |  34    | AURO   | 2023.tejasvini.bachhav@ves.ac.in |
| 4    |Vedanti Tawde     |  28    | AURO   | 2022.vedanti.tawde@ves.ac.in.    |

---

## Guide Details

**Project Guide: Jayshree Ramakrishnan**  
**Department:** Automation and Robotics  
**Institute:** VESIT, Mumbai  

---

## Problem Statement
Existing EMG-based prosthetic and teleoperation control systems rely on a wired 
pipeline — electrodes connected through a microcontroller to a laptop for gesture 
classification — which limits portability and introduces latency and dependency 
on external computing hardware. Additionally, EMG signals from adjacent muscle 
groups often produce overlapping activation patterns for visually distinct 
gestures, leading to classification errors. This project aims to solve both 
problems by using an STM32 microcontroller with on-chip machine learning 
inference and IMU-based orientation sensing to achieve accurate, standalone, 
wireless gesture classification for robotic arm teleoperation.


## Abstract
This project presents a wearable, wireless EMG-based gesture recognition system 
for real-time robotic arm teleoperation. Muscle activity from the bicep and 
tricep is captured using surface EMG electrodes and processed to extract 
envelope, ratio, and co-activation features. An Inertial Measurement Unit (IMU) 
is integrated to capture wrist orientation (roll, pitch, yaw), resolving 
ambiguity between gestures that produce similar EMG activity but differ in 
physical motion. The initial system used an Arduino microcontroller streaming 
data to a laptop for Python-based classification, achieving 82% accuracy with 
six gesture classes. The proposed system upgrades this pipeline to an STM32F4 
microcontroller, leveraging its ARM Cortex-M4F floating-point unit to run a 
trained machine learning model directly on-chip using Edge Impulse and 
STM32Cube.AI, eliminating the need for a connected laptop. Classified gestures 
are transmitted wirelessly via Bluetooth to control a robotic arm. The combined 
EMG and IMU feature set is expected to improve classification accuracy from 82% 
to above 93%, particularly resolving confusion between gestures that share 
similar muscle activation patterns. This system has applications in prosthetic 
control, assistive robotics, and remote teleoperation in hazardous environments.



---

## Objectives
1. To capture and process surface EMG signals from bicep and tricep muscles for 
   gesture classification.
2. To integrate an IMU sensor for wrist orientation data (roll, pitch, yaw) to 
   resolve EMG-only classification ambiguity.
3. To design and train a machine learning gesture classification model using 
   combined EMG and IMU features.
4. To deploy the trained model on an STM32F4 microcontroller for real-time, 
   on-chip inference without dependency on a laptop.
5. To implement wireless (Bluetooth) transmission of classified gestures to a 
   robotic arm for teleoperation.
6. To test and validate system accuracy against the existing Arduino-based 
   baseline (82% accuracy, six gesture classes).


---

## Scope of the Project
- Design and development of a wrist-worn EMG + IMU sensing prototype
- On-device machine learning model training and deployment (Edge Impulse, 
  STM32Cube.AI)
- Wireless communication between wearable unit and robotic arm
- Data collection across six gesture classes (up, down, roll, bicep extend, 
  yaw left, yaw right)
- Performance comparison between EMG-only and EMG+IMU classification accuracy



---

## Existing System
The current baseline system uses an Arduino microcontroller to acquire raw EMG 
signals from AD8232/ExG Pill electrodes, which are streamed over USB to a 
laptop running a Python classifier; the classified gesture is then visualized 
via MATLAB. Limitations of this approach:
- Requires a continuously connected laptop, limiting portability
- Wired USB connection restricts range of motion and mobility
- EMG-only feature set causes classification confusion between gestures with 
  similar muscle activation (e.g., 22% confusion between "up" and "yaw towards 
  right")
- Classification and control logic run off-device, adding latency



---

## Proposed System
The proposed system replaces the Arduino + laptop pipeline with a wrist-worn 
STM32F4 microcontroller unit that performs gesture classification entirely 
on-chip:
- **Main idea**: Combine EMG (muscle activation) and IMU (wrist orientation) 
  features into a single machine learning model, and run that model directly 
  on the microcontroller's hardware floating-point unit for real-time, 
  standalone inference.
- **How it works**: EMG electrodes and a BNO055 IMU feed data to the STM32F4. 
  A model trained via Edge Impulse and converted to optimized C++ using 
  STM32Cube.AI classifies the gesture in under 10ms. The result is transmitted 
  via HC-05 Bluetooth to the robotic arm controller.
- **Major components**: EMG electrodes (bicep/tricep), BNO055 IMU, STM32F4 
  microcontroller (ARM Cortex-M4F with FPU), HC-05 Bluetooth module, robotic 
  arm.
- **Expected benefits**: No laptop dependency, improved classification 
  accuracy (82% → 93%+ expected), lower latency, full portability for 
  wearable/field use.


---

## System Architecture
<img width="784" height="1024" alt="system architecture" src="https://github.com/user-attachments/assets/8da8e836-b06f-458e-9a07-427bdfdf53a5" />

1. EMG sensors — bicep and tricep electrodes capture raw muscle signals
2. Signal processing — baseline correction, rectification, and envelope smoothing clean the raw signal
3. Feature engineering — 11 derived features (ratios, envelopes, co-activation, etc.) extracted per sample

Planned upgrades (teal):
4. Per-user calibration — normalizes signals against each user's max voluntary contraction, so the system works across different body types
5. IMU fusion — adds roll/pitch/yaw from the BNO055 sensor to resolve gesture confusion (e.g. up vs. yaw-right)
6. Gesture classifier — Random Forest / XGBoost model trained on the combined EMG + IMU feature set

Embedded deployment (purple):
7. Edge inference — the trained model runs directly on the STM32F4 chip via Edge Impulse, in under 10ms — no laptop needed
8. Bluetooth link — HC-05 wirelessly transmits the classified gesture

Final output (coral):
9. Robot arm — receives the command and moves in real time, completing the teleoperation loop

In short: muscle signal → cleaned and featurized → calibrated for the individual → fused with motion data → classified → run on-chip → sent wirelessly → robot moves.




---

## Hardware Requirements

|Sr.No.| Component                | Specification                               | Quantity | Purpose                                      |
|------|--------------------------|---------------------------------------------|----------|----------------------------------------------|
| 1.| STM32F411 ("Black Pill")    | STM32F4 series, ARM Cortex-M4F, 100MHz, FPU | 1        | On-chip ML inference microcontroller         |
| 2 | ExG Pill.                   | 3.3V single-channel EMG amplifier           | 2        | Bicep and tricep muscle signal acquisition   |
| 3 | BNO055 IMU                  | 9-axis absolute orientation sensor, I2C     | 1        | Wrist roll/pitch/yaw sensing                 |
| 4 | AD8232                      | ECG/EMG front-end amplifier                 | 1        | Bicep EMG acquisition (current baseline)     |
| 5 | Arduino Uno/Nano            | ATmega328P, USB serial                      | 1        | Signal acquisition (current baseline pipeline)|
| 6 | HC-05 Bluetooth Module      | Class 2, UART, up to 115200 baud            | 1        | Wireless gesture transmission (planned)      |
| 7 | Robotic Arm                 | Servo-driven                                | 1        | Teleoperation output (planned integration)   |

---

## Software Requirements

|Sr.No.| Software / Tool             | Version        | Purpose                                                 |
|------|-----------------------------|----------------|---------------------------------------------------------|
| 1.   | Python                      | 3.x            | Data processing, feature engineering, ML model training |
| 2    | scikit-learn                | latest         | Random Forest / SVM classifier training                 |
| 3    | MATLAB                      | R2023 or later | Robot arm simulation and visualization                  |
| 4    | Edge Impulse Studio         | Web-based      | On-device ML training and STM32 deployment (in progress)|
| 5    | STM32CubeIDE / STM32Cube.AI | latest         | STM32F4 firmware development (planned)                  |
| 6    | Arduino IDE                 | latest         | Baseline firmware (current pipeline)                    |

---

## Technologies Used

* Python (signal processing, machine learning)
* Embedded C (planned STM32 firmware)
* MATLAB (robot arm kinematic simulation/visualization)
* Machine Learning — Random Forest and SVM classification
* Edge Impulse / STM32Cube.AI — on-device ML deployment (in progress)
* Bluetooth (HC-05) — wireless communication (planned)
---

## Methodology

1)Literature survey — Reviewed existing EMG-based control systems, myoelectric prosthetics, and embedded ML deployment approaches (STM32Cube.AI, Edge Impulse) to identify proven signal processing and classification techniques for gesture recognition.

2)Problem identification — Identified three core challenges from initial prototyping: real-time inference latency, EMG confusion between kinematically similar gestures (e.g. flex-up vs. yaw-right), and poor generalization across users with differing muscle mass and electrode placement.

3)Requirement analysis — Defined functional requirements (sub-10ms on-chip inference, six-gesture classification, wireless actuation) and non-functional requirements (subject-independent accuracy, low-cost hardware, real-time operation without a host laptop).

4)System design — Designed the end-to-end architecture: EMG acquisition → signal conditioning → feature engineering → per-user calibration → IMU-fused classification → on-chip inference (STM32F4 + Edge Impulse) → Bluetooth transmission → robotic arm actuation.

5)Hardware/software development — Built the EMG acquisition circuit (AD8232, ExG Pill), implemented the real-time signal processing pipeline in firmware, developed the feature engineering and classification pipeline in Python, and began IMU (BNO055) and STM32F4 integration.

6)Integration — Fused EMG and IMU feature streams, deployed the trained classifier to the STM32F4 target via Edge Impulse, and linked classification output to the robotic arm control interface over Bluetooth (HC-05).

7)Testing and validation — Evaluated classifier accuracy using subject-wise train/test splits to assess cross-user generalization, benchmarked on-device inference latency, and conducted closed-loop tests with multiple subjects performing live gesture-to-actuation control.

8)Documentation and publication — Consolidated results, dataset, firmware, and trained models into the GitHub repository with reproducible setup instructions, accompanied by a final project report and demonstration recording.

---

## Project Timeline

| Week / Month    | Task Planned                                            | Status      |
|-----------------|---------------------------------------------------------|-------------|
| Aug 13 – Aug 24 | Hardware integration — IMU + STM32                      | In Progress |
| Aug 25 – Sep 6  | Calibration & multi-subject protocol design.            | Pending     |
| Sep 6 – Sep 11  | Gesture-to-motion mapping                               | Pending     |
| Sep 11 – Sep 25 | Noise reduction & advanced ML (targeting 90%+ accuracy) | Pending     |
| Oct 3 – Oct 18  | Integration with robot (end-to-end testing)             | Pending     |
| Oct 18 – Oct 30 | System validation and testing                           | Pending     |


## Weekly Progress Updates

Students must update this section every week.

| Week   | Date | Work Completed | Work Planned for Next Week | Issues / Challenges | GitHub Commit Link |
| ------ | ---- | -------------- | -------------------------- | ------------------- | ------------------ |
| Week 1 |      |                |                            |                     |                    |
| Week 2 |      |                |                            |                     |                    |
| Week 3 |      |                |                            |                     |                    |
| Week 4 |      |                |                            |                     |                    |
| Week 5 |      |                |                            |                     |                    |
| Week 6 |      |                |                            |                     |                    |
| Week 7 |      |                |                            |                     |                    |
| Week 8 |      |                |                            |                     |                    |

---

## Design Files

Upload and link all design files here.

| File Type       | File Name / Link | Description |
| --------------- | ---------------- | ----------- |
| Simulation File | main_simulatiom.m| Takes model's predicted gestures from predictions.csv and animates a virtual 4-DOF robot arm in MATLAB |

---

## Circuit Diagram

<img width="1600" height="1011" alt="circuit" src="https://github.com/user-attachments/assets/e28c9597-d222-42b0-8282-29bf5146412e" />

```

---

## Flowchart / Algorithm


### Algorithm
1. Start
2. Initialize sensors (EMG electrodes, IMU)
3. Acquire raw EMG signal (bicep, tricep) and IMU orientation data
4. Process signal — baseline correction, rectification, envelope extraction
5. Compute engineered features (envelope, ratio, co-activation, roll/pitch/yaw)
6. Classify gesture using trained ML model
7. Transmit classified gesture (via Bluetooth, planned) to robotic arm controller
8. Actuate robotic arm according to classified gesture
9. Repeat from step 3

---

## Implementation Details

### Hardware Implementation
**Current baseline:** AD8232 and ExG Pill electrodes are placed on the bicep and
tricep respectively, connected to an Arduino's analog input pins. The Arduino
streams raw voltage readings over USB serial (115200 baud) to a laptop.

**Planned upgrade:** EMG electrodes connect to the STM32F4's ADC pins (PA0,
PA1), and a BNO055 IMU connects via I2C (PB6/PB7) for wrist orientation. An
HC-05 Bluetooth module (UART, PA9/PA10) transmits classified gestures to the
robotic arm controller. Full pinout is documented in `hardware/hardware.md`.


### Software Implementation

Raw serial data is parsed and converted into a labeled, feature-engineered
dataset (`software/prepare_emg_dataset.py`). Features include bicep/tricep
envelope, bicep-to-tricep ratio, and co-activation. A Random Forest classifier
is trained and evaluated (`software/emg_simulation.py`,
`software/emg_classify_export.py`), achieving 81.93% test accuracy. Real-time
classification runs via `software/realtime_inference.py`, and results are
visualized on a simulated robot arm in MATLAB (`software/main_simulation.m`).
For the STM32 upgrade, `software/prepare_for_edge_impulse.py` reformats the
dataset for Edge Impulse Studio, which will generate optimized C++ firmware for
on-device inference via STM32Cube.AI.

---

## Code Structure

```text
EMG_Teleoperation_FinalProject/
│
├── README.md
├── docs/
│   └── literature_survey.md
│
├── hardware/
│   └── hardware.md
│
├── software/
│   ├── software.md
│   ├── prepare_emg_dataset.py
│   ├── prepare_for_edge_impulse.py
│   ├── emg_simulation.py
│   ├── emg_classify_export.py
│   ├── test_accuracy.py
│   ├── realtime_inference.py
│   ├── main_simulation.m
│   ├── emg_model.pkl
│   ├── EMG_TeleOperation_Dataset.csv
│   ├── predictions.csv
│ 
│
├── images/
│   ├── emg_analysis_results.png
│   └── youtube_thumbnail.png
│
└── reference/
    └── paper.md
```

---

## How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/2022khushikandhari-hash/EMG_Teleoperation_FinalProject.git
cd EMG_Teleoperation_FinalProject/software
```

### Step 2: Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost --break-system-packages
```

### Step 3: Run the Classification Pipeline

```bash
python emg_simulation.py          # Train and evaluate the model
python emg_classify_export.py     # Generate predictions.csv for MATLAB
```

### Step 4: Run the MATLAB Visualization

Open MATLAB and run `main_simulation.m` to view the simulated robot arm
responding to classified gestures.

### Step 5 (planned): Prepare Data for Edge Impulse / STM32

```bash
python prepare_for_edge_impulse.py
```
Upload the generated `ei_upload/training/` and `ei_upload/testing/` folders to
Edge Impulse Studio for on-device model training.


## Testing and Results

|Test No.| Test Description                                        | Expected Result                    | Actual Result          | Status  |
|--------|---------------------------------------------------------|------------------------------------|------------------------|---------|
| 1      | Random Forest classifier accuracy on held-out test set  | > 75% accuracy                     | 81.93% accuracy        | Pass    |
| 2      | Per-gesture classification — `up` vs `yaw towards right`| Minimal confusion                  | 22% confusion observed | Fail    |
| 3      | Real-time inference via Arduino serial                  | Correct live gesture classification|                        | Pending |
| 4.     | On-device inference latency (STM32F4, Edge Impulse).    | < 10ms per inference.              |                        | Pending |
| 5.     | End-to-end Bluetooth gesture transmission to robot arm  | Correct robot arm response.        |                        | Pending |

## Result Images / Videos

<img width="2082" height="2281" alt="emg_analysis_results" src="https://github.com/user-attachments/assets/6eaf429d-d6fa-49ed-859b-2c71a644863b" />

```

Video Link:

```markdown
[Project Demo Video](https://drive.google.com/your-video-link)
```

---

## Applications

1. Prosthetic limb control for amputees
2. Assistive robotics for individuals with limited mobility
3. Remote teleoperation in hazardous or inaccessible environments
4. Rehabilitation and physical therapy monitoring

---

## Advantages

1. No physical input device (joystick/keyboard) required — controlled purely by muscle activity
2. Wireless, portable design (once STM32 + Bluetooth upgrade is complete)
3. Combines EMG and IMU data for more accurate gesture classification
4. Low-latency, on-device inference (planned) removes dependency on external computing


---

## Limitations

1. Current baseline still requires a laptop for classification (being addressed by STM32 upgrade)
2. Dataset collected from a single subject — may not generalize across users without retraining
3. Class imbalance in current dataset (`yaw towards right` has ~2x samples of minority classes)
4. `up` gesture has the lowest classification accuracy (~74–76% F1) in the current model

---

## Future Scope

1. Deploy trained model on-chip using STM32F4 + Edge Impulse for standalone operation
2. Explore CNN/LSTM architectures to push accuracy beyond current baseline
3. Multi-subject data collection for a more generalizable model
4. Full integration and end-to-end validation with a physical robotic arm

---

## Research Paper / Publication

| Item                      | Details                                                   |
| ------------------------- | --------------------------------------------------------- |
| Paper Title               |                                                           |
| Conference / Journal Name |                                                           |
| Paper Status              | Not Started / Drafting / Submitted / Accepted / Published |
| Submission Date           |                                                           |
| Paper Link                |                                                           |

---

## References

[1] P. K. Artemiadis and K. J. Kyriakopoulos, "EMG-based teleoperation of a robot arm using low-dimensional representation," in Proc. IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS), 2007.  https://ieeexplore.ieee.org/document/4399452

[2] H. F. Hassan, S. J. Abou-Loukh, and I. K. Ibraheem, "Teleoperated robotic arm movement using electromyography signal with wearable Myo armband," J. King Saud Univ. — Eng. Sci., 2019. https://arxiv.org/pdf/1810.09929

[3] "Robot arm control method using forearm EMG signals," ITM Web Conf., 2020. https://www.researchgate.net/publication/339684308_Robot_arm_control_method_using_forearm_EMG_signals
```

---

## Repository Update Guidelines

Each student team must update the GitHub repository regularly.

Minimum expected updates:

* Update README every week.
* Push code changes regularly.
* Upload circuit diagrams, CAD files, PCB files, reports and presentations.
* Add weekly progress in the progress table.
* Maintain proper folder structure.
* Do not upload unnecessary temporary files.
* Each major update should have a meaningful commit message.

Example commit messages:

```text
Added problem statement and objectives
Updated system architecture diagram
Added sensor interfacing code
Updated weekly progress for Week 3
Added testing results and prototype images
```

---

## Declaration

We declare that this project work is carried out by our team as part of the BE Capstone Project. The work will be regularly updated on GitHub and all references used will be properly cited.

---

## License

This project is for academic use only.

Optional:

```text
MIT License / Creative Commons / Institute Use Only
```

```
```
