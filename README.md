                                        ****Project: 5G Traffic Prediction and Edge Cloud Resource Allocation***

This repository contains three key files:

Traffic Prediction.py: This script implements deep learning and reinforcement learning algorithms to predict 5G network traffic. The code first identifies the best prediction model for each VNF data within 5G network and then uses the Deep Q-Network (DQN) algorithm to dynamically select the best predictor in real time for different traffic types in such networks.

Network Resource Allocation.py: This script focuses on resource allocation within an edge cloud environment. It allocates edge cloud resources proactively, using the traffic predictions generated in Traffic Prediction.py to optimize resource usage.

Proactive Network Resource Allocation.ipynb: This Jupyter notebook contains the full implementation of the project, combining both traffic prediction and resource allocation in one comprehensive solution.
