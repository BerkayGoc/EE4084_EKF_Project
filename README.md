# EE4084: Autonomous Vehicle Lane Following (EKF) Sensor Fusion

## 📌 Project Overview
This repository contains the simulation source code, documentation, and LaTeX materials for the **Autonomous Vehicle Lane Following** project, developed for the EE4084 Kalman and Bayesian Filters course. 

The system implements a multi-rate **Extended Kalman Filter (EKF)** to fuse high-frequency IMU kinematics with lower-frequency GPS and Camera observations. The estimated state drives a closed-loop **Stanley Controller** to achieve stable lane tracking under severe stochastic stress, including total GPS tunnel dropouts and non-Gaussian measurement anomalies.

### 👥 Team Members
* Ahmet Zekeriya Devran (ID: 150721002)
* Berkay Göç (ID: 150721005)
* Boran Varol (ID: 150721023)

---

## 📂 Repository Structure
The repository is organized as follows:

* **`Matlab Code/`**: Contains the core simulation script (`main.m`). This includes the data generation, Kinematic Bicycle Model, EKF predict/update phases, NIS gating logic, and closed-loop control.
* **`LateX/`**: Contains the complete, zipped LaTeX source files and image assets used to compile the initial project proposal and the final academic report.
* **`Docs/`**: Contains the compiled PDF documents (Project Proposal and Final Report) for quick reference.

---

## ⚙️ System Architecture & Technical Specifications

### 1. 6-DOF State Vector ($x_k$)
The system estimates a 6-Degree-of-Freedom state vector, deliberately modeling IMU hardware biases as Wiener processes (random walks) to enable prolonged dead reckoning:

| State Variable | Symbol | Unit | Description |
| :--- | :---: | :---: | :--- |
| **X Position** | $p_x$ | $m$ | Global Cartesian X coordinate |
| **Y Position** | $p_y$ | $m$ | Global Cartesian Y coordinate |
| **Velocity** | $v$ | $m/s$ | Longitudinal forward velocity |
| **Heading** | $\psi$ | $rad$ | Vehicle yaw angle (global) |
| **Gyro Bias** | $\psi_{bias}$ | $rad/s$ | Slow-varying gyroscope thermal drift |
| **Accel Bias** | $a_{bias}$ | $m/s^2$ | Slow-varying accelerometer drift |

### 2. Multi-Rate Sensor Modalities
The EKF handles asynchronous measurements. The Correction phase is dynamically triggered based on the arrival of lower-frequency sensor data, evaluated via **Normalized Innovation Squared (NIS) Gating** to reject severe outliers ($\chi^2$ threshold = 5.991).

| Sensor | Frequency | Observation | Noise ($\sigma$) | Simulated Artifacts |
| :--- | :---: | :--- | :--- | :--- |
| **IMU** | 100 Hz | Kinematics ($a_m, \omega_m$) | $0.5 \ m/s^2$, $0.04 \ rad/s$ | Continuous random-walk drift |
| **Camera** | 30 Hz | CTE, Rel. Heading | $0.20 \ m$, $0.06 \ rad$ | 10% frame drops, 5% glare spikes |
| **GPS** | 10 Hz | Global Position ($x, y$) | $2.0 \ m$ | 9-second tunnel dropout, multipath jumps |

### 3. Core Algorithmic Features
* **Analytic Jacobians:** Deterministic derivation of the process ($F$) and camera observation ($H_{cam}$) Jacobian matrices for optimal covariance propagation.
* **Joseph Form Covariance:** The a posteriori error covariance ($P_{k|k}$) is updated using the numerically stable Joseph form to strictly preserve positive-definiteness under single-precision floating-point operations.
* **Singularity-Protected Stanley Control:** The steering command ($\delta$) is bounded to $\pm35^{\circ}$ and protected against division-by-zero errors at low velocities.

---

## 🚀 Simulation Execution & Results Reproduction

To rigorously evaluate the EKF consistency, robust NIS gating, and closed-loop control performance, follow these straightforward steps to run the simulation locally.

### Prerequisites
* **MATLAB** (R2021a or newer is recommended).
* *Note: The core algorithm is built entirely from scratch to maintain full control over the state matrices. No external black-box toolboxes are required.*

### Execution Steps
1. **Clone the repository:** 
   `git clone [https://github.com/BerkayGoc/EE4084_EKF_Project.git](https://github.com/BerkayGoc/EE4084_EKF_Project.git)`
2. **Navigate to the Source Directory:** Open MATLAB and set your current working directory to the `Matlab Code/` folder.
3. **Run the Core Script:** Execute the `main.m` script from the command window or the editor.

### Expected Output & Evaluation
Once the 80-second multi-rate simulation is executed, the system will provide two layers of evaluation:
* **Command Window Report:** A detailed performance log including Single-Run RMSE metrics, NEES bounds, NIS rejection rates, and a 50-Run Monte Carlo Summary.
* **16 Analytical Figures:** The script will automatically generate report-quality visual data, directly demonstrating:
  * Trajectory dead-reckoning survival during the 9-second GPS tunnel dropout.
  * Real-time Gyroscope and Accelerometer bias tracking within strict $\pm3\sigma$ boundaries.
  * NIS anomaly rejection filtering out severe multipath and glare artifacts.
  * NEES consistency bounds mathematically proving the filter's stability.
