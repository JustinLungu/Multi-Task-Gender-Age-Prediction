# Multi-Task-Gender-Age-Prediction
Performing Multi-Task Learning on Gender and Age features from the UTK Dataset.

This repository contains the codebase for our Machine Learning Practical project at the University of Groningen, where we developed a deep learning model that simultaneously predicts a person’s **gender** and **age** from facial images using **multi-task learning**.

## Overview

Age and gender prediction from facial images is a common computer vision task with applications in fields such as security, biometrics, and demographic analysis. Instead of training separate models for each task, we adopted a **multi-task learning (MTL)** approach to improve efficiency and generalization by sharing learned features between tasks.

Our model uses a **Convolutional Neural Network (CNN)** backbone with **hard parameter sharing**, branching into two task-specific heads:
- **Gender classification** (binary classification)
- **Age prediction** (regression)

The project was trained and evaluated using the **UTKFace** dataset.

## Key Features

- Baseline models for each task separately (gender classification and age regression)
- Multi-task model combining both tasks using a shared CNN
- Data preprocessing pipeline including resizing, grayscale conversion, and augmentation
- Evaluation using appropriate metrics (accuracy, F1 score, MSE)
- Saliency maps for visualizing model attention

## Repository Structure


