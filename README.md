# CortexSort
### Language-Guided Robotic Manipulation with Vision-Language Models
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch MPS](https://img.shields.io/badge/PyTorch-2.4%20(Apple%20Silicon%20MPS)-red.svg)](https://pytorch.org/)
[![RoboSuite](https://img.shields.io/badge/RoboSuite-1.5.1-green.svg)](https://robosuite.ai/)
[![LeRobot](https://img.shields.io/badge/HuggingFace-LeRobot%20SmolVLA-orange.svg)](https://github.com/huggingface/lerobot)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robotics AI project exploring vision-language-guided object perception, task understanding, and autonomous pick-and-place using a Franka Panda robot in PyBullet simulation.

<p align="center">
  <img src="media/One_shot_video_cube_lifting.gif" width="280" alt="Egocentric video human lifting rectangular object" />
  <br>
  <em><b>The Real-World Input</b>: A 6-second egocentric smartphone video of a person reaching out and picking up a steel glass from a table (Vid_0.mp4).</em>
</p>


```html
<p align="center">
  <img src="media/One_shot_video_cube_lifting.gif" width="280" alt="First-person footage of a person lifting an object from a table" />
  <br>
  <em><b>Human Demonstration:</b> A short, 6-second smartphone recording captured from chest height, showing a person reaching for and lifting a steel cup from a tabletop (Vid_0.mp4).</em>
</p>
```

### 💡 Project Goal

**Can a robot learn to perform a physical task by observing a short video of human doing it?**

This project investigates how everyday video recordings can be used to teach robots manipulation skills. Starting with a first-person recording of a human picking up a rectangular box, the pipeline connects human motion analysis, robot simulation, and learning-based control.

The workflow consists of three main components:

1. **Recovering Human and Object Motion:** Analyze the video to estimate how the hand and target object move through space, using visual information rather than dedicated motion-capture systems, body-worn sensors, or physical tracking markers.

2. **Transferring Demonstrations to a Robot:** Convert the estimated human movements into a representation suitable for a 7-degree-of-freedom Franka Emika Panda robot. RoboSuite and MuJoCo provide the simulated environment for developing and testing the resulting manipulation behavior.

3. **Learning a Visual Control Policy:** Work with SmolVLA, a vision-language-action model with approximately 450 million parameters, to investigate how a robot can use camera observations to guide its actions and reproduce the demonstrated task.

The broader aim is to explore a more accessible approach to robot skill acquisition, where human videos provide demonstrations that can help train robots to perform manipulation tasks in simulation.

---
