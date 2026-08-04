# FNO for Battery Thermal Dynamics
### Fast, Resolution-Invariant Prediction of Thermal Runaway

---

## Slide 1: The Real-World Problem
*   **Current State:** Battery Thermal Runaway is predicted using Finite Element Analysis (FEA) or physical testing.
*   **The Issue:** FEA calculates heat transfer at every corner, microsecond by microsecond. It is highly accurate but incredibly slow and computationally expensive.
*   **The Goal:** Predict battery failure instantly to save millions in testing time and prevent fires in production pipelines.

---

## Slide 2: The Solution - Fourier Neural Operators
*   Standard AI fails at physics, and PINNs are too slow because they have to be retrained for every new starting condition.
*   **FNOs operate in the Frequency Domain:** They use the Fourier Transform to look at the global frequencies of the heat wave.
*   **Resolution-Invariant:** You can train the AI on a low-res heat grid and use it to predict on a high-res grid without retraining.
*   **Speed:** 1000x faster than traditional FEA because it can "fast-forward" time in one mathematical step.

---

## Slide 3: Code Architecture (PyTorch)
*   **Input:** 2D spatial grid of battery temperature sensors.
*   **Transform (`torch.fft`):** Moves the data from spatial domain to frequency domain.
*   **Spectral Convolution:** Multiplies complex weights to capture smooth, low-frequency physics while filtering out high-frequency noise.
*   **Output (`torch.ifft`):** Moves data back to physical space, bypassing traditional CNN bottlenecks.

---

## Slide 4: Visualizing the Results
*(Insert `results/evaluation_sample_1.png` here)*

*   **Left Image:** Initial Hot Spot (Thermal anomaly detected).
*   **Middle Image:** Ground Truth Diffusion (What traditional simulation says will happen).
*   **Right Image:** FNO Prediction (What our AI predicted instantly).
*   **Conclusion:** The FNO successfully learned the underlying Heat Equation PDE!
