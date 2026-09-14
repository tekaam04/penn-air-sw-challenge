# PennAiR Software Challenge (Fall 2026)

Last year, as part of the SAE Aero Design Competition, we had to come up with a vision and control algorithm to autonomously detect and land within a marked region on a field. Now, you guys can try a simplified version yourselves!

## Objective

Develop an algorithm to detect solid shapes on a grassy background, trace their outlines, locate their centers, and apply the algorithm to a video. For extra credit, make the algorithm agnostic to background color and shape color gradients. You should attempt to get as far in the challenge as you can, but we are not necessarily expecting all parts to be completed.

## Task Overview

### Part 1: Shape Detection on Static Image

- Use [this image (PennAir 2024 App Static.png)](https://drive.google.com/file/d/10jl6ac0LPlVkLKmZWh0SU-7O4BoKZOkH/view?usp=sharing) that features solid shapes on a grassy background.
- Implement an algorithm (we recommend OpenCV — `cv2`) to detect the shapes.
- Trace the outlines of the detected shapes.
- Locate and mark the centers of the shapes.

### Part 2: Shape Detection on Video

- Apply the algorithm to the video file [PennAir 2024 App Dynamic.mp4](https://drive.google.com/file/d/1ERhZpzPPjTeq6dOluvXXefJGqdVbnF-t/view?usp=sharing).
- Ensure the algorithm consistently detects, traces, and locates the centers of the shapes throughout the video.
- Please treat the video file as a streamed input — this part should involve applying your prior algorithm to each frame. Just like how our aircraft doesn't have the entire video journey ahead of time, we want to feed the algorithm each frame one at a time.

### Part 3: Background Agnostic Algorithm

- Modify the algorithm to work with various background colors and textures.
- Ensure the algorithm maintains its accuracy in detecting, tracing, and locating shapes regardless of the background.
- Test with [PennAir 2024 App Dynamic Hard.mp4](https://drive.google.com/file/d/1pdNlsmfkLPYwAr3T7Noi_Gb-TKHZWLv_/view?usp=sharing).

> **NOTE:** For parts 1–3, you should be creating your own algorithm. Please avoid using pretrained models or existing end-to-end solutions that can solve the task out of the box. Using models is totally fine if you've trained or fine-tuned them yourselves. We want to see what creative solutions you guys come up with yourselves!

### Part 4: Make it 3D

You may have realized that this has been a 2D task, but the task should actually be in 3D. Extend your algorithm to instead mark the depth, x, and y coordinates of the centers w.r.t. the camera. Assume that the camera's [intrinsic matrix](https://ksimek.github.io/2013/08/13/intrinsic/) is

```
K = [[2564.3186869,  0,             0],
     [0,             2569.70273111, 0],
     [0,             0,             1]]
```

and that the circle has a radius of 10 in. For simplicity, assume a flat surface.

### Part 5: Let's add this to ROS2 (optional, highly recommended)

- Create a node that streams the input video as images.
- Create a node that runs your shape detection algorithm on those images.
- Publish detections (positions, outlines) on a ROS2 topic.
- Provide a launch file to run the system.
- Hint: read the [docs](https://docs.ros.org/en/jazzy/Concepts/Basic.html).

> **NOTE:** If using Windows, the easiest way to install ROS is either through WSL2 Ubuntu, or running dual-boot Ubuntu. We recommend using Ubuntu 24.04 with ROS 2 Jazzy, but any recent combination of Ubuntu/ROS versions should work (22.04/Humble, 26.04/Lyrical).

> **NOTE:** For macOS users, it is recommended to set up an Ubuntu VM via UTM. [Installation](https://docs.getutm.app/installation/macos/) – [Ubuntu Setup](https://docs.getutm.app/guides/ubuntu/)

### Part 6: Anything else

If you have any improvements/extra enhancements in mind (tracking objects even through overlap, etc.), implement them! We're excited to see what you come up with.

For reference, here is an example of something that accomplishes everything up to Part 3: [pennair_2024_sample sol.mov](https://drive.google.com/file/d/1v059K4DKT4FO-9bJceoaWdYjD5aPGbyl/view?usp=sharing)

## Deliverables

### Code Implementation

- Provide the source code for the implemented algorithm.
- The code should be well-(enough)-documented and include comments explaining the key steps and logic.

### Static Image Results

- Submit the processed image showing detected shapes with their outlines traced and centers marked.
- Provide a brief explanation of your approach and any challenges faced.

### Video Results

- Submit a screen recording of the processed video, demonstrating the algorithm's performance.
- Include a brief report on how the algorithm performs and any adjustments made to improve it (e.g. using less computationally-expensive operations).

### Background Agnostic Results

- Again, the processed video demonstrating the algorithm's performance.
- Include a brief discussion on the modifications made to achieve background agnosticism.

## Submission Instructions

### Repository

- Create a GitHub repository for your project.
- Include all source code, processed images, videos, and reports in the repository.
- Ensure the repository is public and share the link in the Google Form field.

### Documentation

- Include a `README.md` file in the repository with clear instructions on how to run your code.
- Provide any additional documentation required to understand and evaluate your solution.
- Addendum: it would be greatly appreciated if you could also embed the videos in your README file(s) 🙂

## Evaluation Criteria

- **Accuracy:** Precision in detecting, tracing, and locating shapes.
- **Efficiency:** Performance of the algorithm in real-time processing for videos.
- **Robustness:** Ability to handle different backgrounds and maintain accuracy.
- **Code Quality:** Clarity, documentation, and organization of the code.
- **Innovation:** Creative approaches to solving the challenge and handling potential issues.
- **Effort & AI Usage:** AI tools are allowed to help you during development and for understanding new concepts, but your submission should reflect your own understanding, effort, and problem-solving. If it is clear that the solution was largely AI-generated with minimal effort or understanding from the applicant, this will be taken into consideration during evaluation.

---

We look forward to seeing your innovative solutions and considering you for PennAiR's software team!

Questions: email board@pennaerial.com or come to office hours in Towne M81.
