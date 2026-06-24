# Edge AI on Microcontrollers: Real-Time Gesture Recognition via Quantized CNN on STM32H7 🚀

[![STM32H7](https://img.shields.io/badge/Hardware-STM32H745ZITx-blue.svg)](https://www.st.com/)
[![TinyML](https://img.shields.io/badge/Field-TinyML%20%2F%20Edge%20AI-rebeccapurple.svg)]()
[![Toolchain](https://img.shields.io/badge/Toolchain-STM32CubeIDE%20%2F%20X--CUBE--AI-orange.svg)]()

This repository contains the implementation of an embedded **Edge AI** system for real-time gesture recognition of the Rock-Paper-Scissors game. The project is deployed on the **STM32H745ZI-Q** development board using an **OV7670** camera sensor.

---

## 🎯 Project Objectives
To demonstrate the feasibility and accuracy of embedded visual recognition by running the entire pipeline—acquisition, pre-processing, and inference—directly on the microcontroller, while adhering to the following constraints:
* **Autonomy:** No offloading to external servers or Cloud infrastructures. All computations are performed *on-device*.
* **Responsiveness:** Total inference latency targets under **100ms** (achieved approximately **89.44 ms** during benchmarking).

---
## 📺 Demo Video
<img width="800" height="363" alt="video_gioco_morra_cinese-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/649eb465-778a-46c8-85d4-3c3de0764345" />

---
## 💻 Target Hardware & Technical Specifications

### 1. Microcontroller: STM32H745ZI-Q
* **High-Performance Dual-Core Architecture:**
  * **Cortex-M7 (480 MHz):** Main high-performance core entirely dedicated to running the Convolutional Neural Network (CNN).
  * **Cortex-M4 (240 MHz):** Disabled during this initial phase to accurately map power consumption and optimize the primary core's performance.
* **Memory Management:** 2 MB Flash and 1 MB total segmentable RAM.

### 2. Image Sensor: OV7670
* **Maximum Resolution:** VGA ($640 \times 480$ pixels), configured to a reduced resolution to maintain a low memory footprint.
* **Frame Rate:** Up to 30 fps at full VGA resolution.
* **Operating Voltage:** 3.3V DC.
* **Supported Output Formats:** Raw RGB, RGB (RGB565/RGB555), YUV (4:2:2), and YCbCr (4:2:2). In this project, it is configured for **QVGA ($320 \times 240$)** capture in **YUV (4:2:2)** format.
* **Control Interface:** Serial protocol similar to **I2C (SCCB)** for internal register configuration (Gain, AWB, exposure).

---

## 🔄 Workflow

```text
+-----------------------+      +--------------------------+      +--------------------------+      +---------------------------+
| 1. Data Eng & Train   |      | 2. Hardware Config       |      | 3. Model Ingestion (AI)  |      | 4. Firmware Application   |
| (Python / Keras / TF) | ---> | (STM32CubeMX)            | ---> | (STM32Cube.AI Studio)    | ---> | (STM32CubeIDE / C / HAL)  |
| Kaggle DS -> .tflite  |      | DCMI, DMA, Clock Config  |      | PTQ INT8 -> Librerie C   |      | Driver OV7670, Cache, ISR |
+-----------------------+      +--------------------------+      +--------------------------+      +---------------------------+
```

## 🔄 Game Flow & Execution Pipeline

The logical and operational pipeline of the game is divided into 7 distinct phases:
```text
[Phase 1: Blue Button] ──> [Phase 2: Countdown] ──> [Phase 3: Random MCU Move]
│
[Phase 6: Referee Logic] <── [Phase 5: AI Inference] <── [Phase 4: DCMI/DMA Capture]
│
└──> [Phase 7: PC Feedback & Stream]
```
1. **Phase 1: User Input (Blue Button)**
   The match is triggered by pressing the onboard blue user button (**Pin PC13**). Input is handled via an external interrupt (**EXTI Line 13**) on the rising edge, routed to the Cortex-M7 core.
2. **Phase 2: Countdown**
   The MCU sends a text-based countdown (`INFO: Get ready...`, `COUNT: 3`, `COUNT: 2`, `COUNT: 1`) via **USART3** to the PC at ~300ms intervals, giving the user time to position their hand.
3. **Phase 3: MCU Move (Honest Random Selection)**
   To guarantee absolute system fairness, the board's move is calculated and isolated *before* capturing and processing the user's hand image. The calculation uses: `stm32_move = HAL_GetTick() % 3;` which deterministically maps to: `0` (Paper), `1` (Rock), `2` (Scissors).
4. **Phase 4: Image Capture**
   As the countdown ends, the hardware **Snapshot mode** is triggered using the **DCMI** (*Digital Camera Interface*). Pixels are written directly into the RAM Frame Buffer via the **DMA** controller, bypassing the CPU. An asynchronous notification is generated upon completion via `HAL_DCMI_FrameEventCallback`.
5. **Phase 5: AI Inference & Neural Network**
   * **Model:** Based on the **MobileNetV2** architecture, optimized by Google for resource-constrained mobile and embedded environments.
   * **Training:** Implemented via **Transfer Learning** using a Kaggle dataset (`sanikamal/rock-paper-scissors-dataset`), freezing the base weights pre-trained on ImageNet.
   * **Optimization (INT8 Quantization):** Utilizing the **STM32Cube.AI Studio** tool, the model was post-training quantized to 8-bit integers. This achieved a 4x reduction in Flash memory footprint.
   * **On-Target Resource Metrics:**
     * **Inference Time:** ~89.44 ms (~11.18 inferences/sec).
     * **Flash Usage:** ~404 KB (Weights: 404 KB).
     * **RAM Usage:** ~225 KB (Activations: 225 KB).
     * **Model Input Dimensions:** 128x128x1 (8-bit grayscale).
6. **Phase 6: Referee Logic**
   The comparison logic is executed locally to declare the winner (e.g., Rock beats Scissors, Scissors beats Paper, etc.). If identical moves are detected, a tie is declared.
7. **Phase 7: Transmission & PC Feedback**
   Game data is transmitted via serial communication upon match completion:
   * **Text Logs (USART3):** Transmission of the selected moves and the final match verdict.
   * **Binary Image Stream:** Sequential transfer of the 16,384 bytes ($128 \times 128$ pixels) making up the processed frame.
   * **PC Interface:** A Python script on the PC receives the raw stream, reconstructs the frame, and displays the graphical result on screen.

---

## 🔮 Future Enhancements & Roadmap
* **Data Augmentation:** Introducing complex and heterogeneous backgrounds into the training dataset to increase inference robustness in unpredictable real-world environments.
* **Visual Robustness:** Target training with samples featuring harsh shadows and variable lighting conditions.
* **Dual-Core Optimization:** Offloading the I/O management and DCMI peripheral tasks entirely to the secondary **Cortex-M4** core, leaving the high-performance **Cortex-M7** dedicated solely to the heavy AI model inference.
* **Hardware Upgrade:** Replacing the OV7670 sensor with a camera module offering higher light sensitivity and superior noise tolerance.
