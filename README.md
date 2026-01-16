# ⚡ CUDA_0: The Acceleration Manifesto

![NVIDIA](https://img.shields.io/badge/Architecture-NVIDIA%20CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Language](https://img.shields.io/badge/Code-Python-blue?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active%20Development-success?style=for-the-badge)
![Hardware](https://img.shields.io/badge/GPU-RTX%2030%2F40%2F50%20Ready-orange?style=for-the-badge&logo=geforce&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

> *"RAM prices are soaring. Games are $70. Your GPU is a supercomputer waiting to be unleashed. Stop consuming pixels—start computing them."*

## 🏴‍☠️ Mission: Reclamation

Welcome to **CUDA_0**. This repository is a collection of high-performance, GPU-accelerated projects designed to push consumer hardware to its absolute mathematical limits.

We are living in an era where "beefy machines" are marketed solely for gaming, while their compute capability remains dormant 90% of the time. This repo is the antidote. We reject the notion of buying hardware just to play; we buy it to **crunch**.

From simulating relativistic astrophysics to training neural networks from scratch, **CUDA_0** is about unlocking the raw TFLOPS sitting under your desk.

---

## 🌌 Flagship Project: Gargantua (Hyper-Accreted)

**Current Status:** *Live & Optimized*

A 4K-capable, CUDA-accelerated simulation of a Schwarzschild black hole. It leverages parallel kernels to compute relativistic physics, gravitational lensing, and Doppler beaming for over **40,000 particles** in real-time.

* **Tech Stack:** Python, Numba, Pygame
* **Performance:** ~145 FPS @ 1920x1200 (RTX 50-Series)
* **Key Feature:** Moving from O(n) CPU loops to O(1) parallel GPU execution.

[📂 **Explore the Code**](./Proj01Gargantua_Hyper-Accreted) *(Link to your folder)*

---

## 🔮 The Roadmap: Future Accelerations

This repository will grow into a library of parallel computing experiments. Upcoming modules will focus on:

### 1. 🧮 Matrix & Tensor Engines
- Building custom BLAS (Basic Linear Algebra Subprograms) kernels.
- Implementing Matrix Multiplication algorithms (Tiled, Shared Memory) that rival cuBLAS.

### 2. 🪐 N-Body Physics
- Simulating star clusters or galaxy collisions.
- Moving from $O(N^2)$ complexity to optimized Barnes-Hut algorithms on the GPU.

### 3. 🕸️ Neural Computing
- Implementing "Forward Pass" and "Backpropagation" purely in CUDA kernels.
- Building a perceptron from scratch without high-level frameworks like PyTorch.

### 4. 🌊 Fluid Dynamics (CFD)
- Real-time Lattice Boltzmann simulations.
- Visualizing smoke and water flow using GPU grid solvers.

### 5. 🔦 Ray Tracing (RTX)
- Writing a path tracer from scratch.
- Utilizing RT Cores for non-gaming applications.

## *And Many more to come..............*

---

## 🛠️ General Environment Setup

To run most projects in this repo, you will need a standardized "Compute" environment.

**The "Golden Standard" Conda Setup:**
```bash
# 1. Create the environment
conda create -n cuda_0 python=3.10
conda activate cuda_0

# 2. Install the Core Engine (Toolkit & Numba)
conda install cudatoolkit numba numpy scipy

# 3. Install Visualization & System Tools
pip install pygame matplotlib
conda install pywin32  # Windows Hardware Access
```

## 🤝 Contributing
Got a "beefy machine" gathering dust?

- Fork this repo.
- Clone your fork.
- Optimize a kernel or add a new simulation.
- Push and create a Pull Request.

Note: We prioritize **speed**,**memory efficiency**, and **raw math**. If it runs on the CPU, it doesn't belong here.

## 📜 License
This project is open-source under the MIT License. Use the code. Fork the physics. Burn the GPU.

---
## Aman Banik ([AmanBaniksR06](https://github.com/AmanBanik)) ✌️