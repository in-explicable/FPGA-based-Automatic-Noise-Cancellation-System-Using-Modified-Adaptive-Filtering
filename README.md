# FPGA-Based Optimized Automatic Noise Cancellation System

This repository contains the implementation and documentation for the project **"FPGA-Based Optimized Automatic Noise Cancellation System Using Modified Adaptive Filtering"**. The project leverages advanced algorithms such as Recursive Least Squares (RLS), Spectral Subtraction, and Particle Swarm Optimization (PSO) on the Xilinx PYNQ-ZU FPGA platform.

---

## Project Overview
This project aims to address the limitations of traditional noise cancellation systems by implementing a highly efficient adaptive filtering system on FPGA. The core features include:
- Low-latency and high-throughput noise cancellation.
- Real-time processing enabled by the PYNQ-ZU platform.
- Optimization of adaptive filter parameters using PSO for faster convergence.

---

## Features
- **RLS Algorithm**: Fast convergence for noise cancellation.
- **Spectral Subtraction**: Preprocessing to handle stationary noise.
- **PSO**: Dynamic tuning of adaptive filter parameters.
- **FPGA Implementation**: Real-time processing with minimal latency.

---

## Hardware and Software Requirements
### Hardware:
- Xilinx PYNQ-ZU FPGA Development Board.
- Audio input/output peripherals (microphone, headphones).

### Software:
- Vivado HLS and Vivado Design Suite.
- MATLAB R2023a or later for simulations.
- PYNQ Framework for FPGA integration.
- Python for real-time Jupyter Notebook execution.

---

## System Architecture
The system consists of the following components:
1. Noise preprocessing using **Spectral Subtraction**.
2. Noise cancellation using the **RLS Algorithm**.
3. Optimization of filter parameters with **PSO**.
4. Real-time processing facilitated by the **PYNQ-ZU platform**.

---

## Results
- **SNR Improvement**: Significant noise reduction compared to LMS and NLMS algorithms.
- **Latency**: Low latency suitable for real-time applications.
- **Resource Utilization**: Efficient use of FPGA resources, including logic elements, DSP blocks, and memory.

## Future Scope
- Extension to multichannel noise control systems.
- Integration of machine learning models for adaptive noise filtering.
- Optimization of FPGA designs for further energy efficiency.
- Testing in real-world environments for broader applicability.

## Contributors
- Nivesh S
- Priyadharshini
- Kiruthik Shankar
- Gaurav Kumar

Guided by: **Dr. A.V. Ananthalakshmi**

Department of Electronics and Communication Engineering, Puducherry Technological University.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
