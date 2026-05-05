# Phase 11R — Threshold Recalibration

## Objective
Re-evaluate threshold probabilities after implementing `SMOTENC` to find the most balanced decision boundary for both the 0.3 and 1.0 models.

## Input State
- Models trained utilizing `SMOTENC` instead of standard `SMOTE`.
- Class weights were removed, changing the underlying baseline prediction probabilities.

## Actions Taken
1. Re-evaluated threshold points from 0.1 to 0.9 across both `SMOTENC 0.3` and `SMOTENC 1.0` prediction arrays.
2. The removal of class weights shifted the global average prediction probability downward for both models, keeping the optimal threshold naturally lower than 0.5.
   - For `SMOTENC 0.3`, Threshold `0.2` produced the best F1-Score balance.
   - For `SMOTENC 1.0`, Threshold `0.1` produced the best balance.

## Interpretation Logic
"After correcting the sampling strategy to strictly respect categorical boundaries via `SMOTENC`, prediction confidence shifts. The removal of class weights means the network relies purely on the physical sample distribution, shifting optimal thresholds dynamically."
