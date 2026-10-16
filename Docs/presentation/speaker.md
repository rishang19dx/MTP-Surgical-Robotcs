# CS399P Mid-Evaluation: Final Speaker Notes & Defense Guide

**Candidate:** Rishang Yadav (Roll No: B23173)  
**Supervisor:** Dr. Deepak Raina  
**Department:** School of Computing and Electrical Engineering (SCEE), IIT Mandi  
**Presentation Files:** `template.tex` (Custom Class) / `presentation_standalone.tex` (Standard Madrid Theme)  
**Total Slides:** 13 Slides (16:9 Widescreen)

---

## Presentation Overview & Defense Strategy

This guide provides the exact spoken dialogue for each slide. Each talk track is designed to:
1. **Preempt Obvious Panel Questions:** Resolves panel doubts directly within your flow of speech before they can interrupt.
2. **Use Simple, Human Language:** Zero buzzwords or AI slop—everything is grounded in clear engineering logic.
3. **Walk Through Flowcharts Intuitively:** Explains Slide 7 (Surgical Bridge) and Slide 8 (3D Ultrasound Pipeline) so clearly that any professor—regardless of their research area—can follow along immediately.

---

## Slide 1: Title Slide
* **Slide Title:** Simulation Platform & Autonomous Framework for Next-Generation Medical Robotics
* **Spoken Dialogue:**
  > "Respected committee members and Dr. Raina, good morning. Welcome to my mid-term evaluation for CS399P. My project focuses on developing a simulation platform and autonomous 3D ultrasound scanning framework for medical robotics.  
  > Over the first two months, our goal was to build the core engineering foundation: specifically, a zero-lag physics bridge connecting PyBullet to clinical 3D Slicer, and a GPU-accelerated 3D ultrasound reconstruction pipeline. Today, I will walk you through our motivation, what we have built so far, and our roadmap for AI policy benchmarking."
* **Panel Questions Cleared in Dialogue:**
  * *What milestone is this?* Mid-term evaluation of CS399P (first two months of work).
  * *What is the scope so far?* Foundation and simulation platform development, setting up future AI benchmarking.

---

## Slide 2: Motivation: The Healthcare AI Dilemma
* **Slide Title:** Motivation: The Healthcare AI Dilemma
* **Spoken Dialogue:**
  > "To understand our motivation: why can't we just collect training data on physical robots like people do in warehouse automation? In factory robotics, you can run a robot 24/7 without safety concerns. In surgery, you cannot experiment on living human patients—it is dangerous, expensive, and difficult to standardize across different body shapes.  
  > We chose robotic ultrasound as our benchmark because it presents a unique physical challenge: unlike optical cameras where pulling back 2 cm still gives you an image, ultrasound requires steady physical contact. If your probe angle or contact force is even slightly off, acoustic decoupling occurs and the screen goes completely black. If you push too hard, you cause patient pain and deform the organ.  
  > Because imaging and physical contact are so tightly linked, simulation is not just a convenience—it is the mandatory pre-condition for training any medical robot policy."
* **Panel Questions Cleared in Dialogue:**
  * *Why not just collect real-world data?* Ethically impossible on patients; hard to standardize.
  * *Why benchmark on ultrasound instead of optical surgery?* Ultrasound signal is completely coupled to physical contact force; losing contact yields complete noise.

---

## Slide 3: Project Objectives
* **Slide Title:** Project Objectives
* **Spoken Dialogue:**
  > "To keep our project focused, our work is split into two clear, connected objectives:  
  > In **Objective 1**, we build the simulation platform. Because training robots on real patients is impossible, we need a virtual environment that models probe pressure on soft tissue at 1000 Hz, generates live ultrasound images with realistic wave shadows directly from CT scans, streams directly into hospital software (3D Slicer), and enforces virtual keep-out boundaries around sensitive organs.  
  > In **Objective 2**, we use this simulator as a safe training ground. We teach robots how to scan autonomously—moving the probe intelligently to build complete 3D scans and look around bone shadows—and benchmark how well different AI models perform.  
  > As the arrow in the diagram shows, Objective 1 creates the realistic training data that Objective 2 needs to teach autonomous scanning algorithms. The first two months delivered the Objective 1 platform and formulated the active reconstruction algorithm."
* **Panel Questions Cleared in Dialogue:**
  * *Is this a robotics or an AI project?* Both: Objective 1 builds the robotics simulator, which generates the data to train and benchmark AI policies in Objective 2.
  * *What was done in the first 2 months?* Objective 1 platform development and the active scanning formulation.

---

## Slide 4: Literature Survey (1/2): Robotic Ultrasound & Simulators
* **Slide Title:** Literature Survey (1/2): Robotic Ultrasound & Simulators
* **Spoken Dialogue:**
  > "When surveying existing ultrasound simulators, our primary baseline was **SonoGym (Ao et al., 2025)**. Built on NVIDIA IsaacLab, SonoGym demonstrated GPU-accelerated reinforcement learning for probe tracking. However, it has four critical limitations that prevent realistic training:  
  > First, it explicitly assumes an external controller glues the probe to the skin. It only plans 3-DoF sliding motion along the surface, completely bypassing normal contact force mechanics.  
  > Second, its patient model is rigid with zero soft-tissue deformation.  
  > Third, it uses a 2D pix2pix GAN that cannot model physical sound wave attenuation or off-axis bone shadows.  
  > And fourth, it operates headlessly without any integration with clinical tools like 3D Slicer.  
  > On the visualization side, existing bridges like **SlicerROS2** proved the clinical value of 3D Slicer, but running heavy ROS2 middleware creates serialization lag and complex setup on hospital workstations. These gaps defined our engineering goals."
* **Panel Questions Cleared in Dialogue:**
  * *Does SonoGym use PyBullet?* No, SonoGym uses NVIDIA IsaacLab (PhysX). We use PyBullet in our bridge for 1000 Hz contact dynamics.
  * *Why not just use SonoGym directly?* It bypasses normal contact force physics, uses rigid patient anatomy, has no 3D Slicer link, and lacks wave-based ray acoustics.

---

## Slide 5: Literature Survey (2/2): Surgical Learning & World Models
* **Slide Title:** Literature Survey (2/2): Surgical Learning & World Models
* **Spoken Dialogue:**
  > "Next, we surveyed surgical AI models. Recent works like **Surgical Robot Transformers (SRT and SRT-H)** demonstrated that diffusion policies and action chunking produce remarkably smooth robot trajectories under language control. However, they tested strictly on rigid peg-transfer and needle manipulation tasks without continuous contact or ultrasound imaging.  
  > Meanwhile, **Surgical World Models like Cosmos-H** demonstrate the appeal of video diffusion. But pure video generation models can hallucinate anatomical boundaries because they lack physical collision constraints, and their inference latency is too high for real-time robot control loops.  
  > This convinced us that the right approach is a hybrid one: grounded physics simulation and ray acoustics combined with generative action policies."
* **Panel Questions Cleared in Dialogue:**
  * *What about modern foundation models and world models?* They work well for optical manipulation, but hallucinate boundaries and lack contact force feedback required for ultrasound.

---

## Slide 6: What We Are Doing Differently Than Baselines
* **Slide Title:** What We Are Doing Differently Than Baselines
* **Spoken Dialogue:**
  > "To be clear, existing baselines are not wrong—they were designed for different priorities. This slide illustrates how we adapt established paradigms to our clinical setting:  
  > First, for the clinical interface: ROS2 is standard across general robotics, but on hospital computers it adds unnecessary multi-node daemon overhead. Because 3D Slicer natively supports OpenIGTLink, direct TCP binary sockets give us lag-free 60 FPS streaming without middleware bloat.  
  > Second, for ultrasound generation: Data-driven 2D GANs produce realistic textures, but they require retraining for each patient and cannot compute sound attenuation. Our GPU ray shader simulates real wave mechanics zero-shot directly from patient CT scans.  
  > Third, for planning: Rather than 2D visual tracking or blind zigzag sweeps, our framework reconstructs a full 3D volume with uncertainty maps and actively plans next-best views to eliminate acoustic shadows."
* **Panel Questions Cleared in Dialogue:**
  * *Are you claiming baselines are defective?* No, baselines have different design trade-offs. We adapt them for hospital workstation compatibility and wave physics.
  * *Why avoid ROS2?* 3D Slicer natively speaks OpenIGTLink. Direct TCP sockets eliminate serialization latency and multi-node setup.

---

## Slide 7: Work Accomplished: Slicer-PyBullet Surgical Bridge
* **Slide Title:** Work Accomplished: Slicer-PyBullet Surgical Bridge
* **Spoken Dialogue:**
  > "Here is our first completed technical milestone: the Slicer-PyBullet Surgical Bridge. Even if you haven't seen this pipeline before, the diagram makes the architecture very simple:  
  > On the right, **PyBullet** acts as our virtual operating room. It simulates the robot arm and soft tissue contact dynamics at 1000 Hz so forces stay physically stable.  
  > On the left, **3D Slicer** serves as the surgeon's clinical viewport, rendering the patient and robot at 60 FPS.  
  > In the middle, the **OpenIGTLink protocol** connects them over direct TCP sockets: sending clinician target points and safety limits to the simulator, and streaming live robot kinematics, tissue meshes, and contact force vectors back to the screen.  
  > Three key engineering decisions make this work:  
  > 1. **Decoupled Multi-Rate Timing:** Physics runs at 1000 Hz while display runs at 60 FPS. If the 3D graphics hiccup, physics never crashes.  
  > 2. **In-Place Memory Updates:** We overwrite existing graphics point buffers directly in memory rather than deleting and recreating 3D objects, eliminating GUI stutter and memory leaks.  
  > 3. **Virtual Safety Fixtures:** If a surgeon outlines a sensitive organ on the CT scan in Slicer, PyBullet enforces it as an impenetrable physical wall."
* **Panel Questions Cleared in Dialogue:**
  * *How do Slicer and PyBullet communicate?* Via OpenIGTLink binary TCP packets, with multi-rate clock decoupling (1000 Hz physics $\leftrightarrow$ 60 FPS GUI).
  * *What is an 'in-place update'?* Directly writing new 3D point positions into pre-allocated VTK C++ memory buffers instead of destroying and re-instantiating scene graph nodes.

---

## Slide 8: Work Accomplished: 3D Ultrasound Reconstruction Pipeline
* **Slide Title:** Work Accomplished: 3D Ultrasound Reconstruction Pipeline
* **Spoken Dialogue:**
  > "Our second technical milestone is the 3D Ultrasound Reconstruction Pipeline. Let's trace the closed loop shown in the diagram step-by-step:  
  > 1. We start with the **Patient CT Scan**—a 3D density tensor.  
  > 2. At the robot's current probe pose, our **GPU Reslicer** cuts a 2D slice using trilinear interpolation in under 2 milliseconds.  
  > 3. Our **Acoustic Shader** simulates physical sound wave decay and speckle noise, producing an authentic B-mode ultrasound image.  
  > 4. The **Confidence Map** identifies acoustic shadow zones where sound cannot penetrate bone.  
  > 5. The **3D Voxel Fusion** module uses the robot's tracked arm poses to stitch these 2D frames into a global 3D voxel volume.  
  > 6. The **Uncertainty Map** spots dark, unobserved regions.  
  > 7. The **Active View Planner** computes the best next angle to peer around bone obstacles.  
  > 8. Finally, the **Robot Controller** solves inverse kinematics to move the 7-DoF arm to that new pose, completing the loop.  
  > This gives us a closed-loop system where the robot actively resolves its own acoustic shadows."
* **Panel Questions Cleared in Dialogue:**
  * *How is ultrasound simulated in real time? Isn't wave physics slow?* Full wave equations are slow, but our GPU PyTorch shader computes trilinear slicing and exponential ray attenuation in under 2 milliseconds.
  * *How are 2D slices combined into 3D?* Tracked robot forward kinematics place each local 2D scan into a global 3D voxel grid with confidence weighting.

---

## Slide 9: Autonomous Scanning: Formulating Active View Selection
* **Slide Title:** Autonomous Scanning: Formulating Active View Selection
* **Spoken Dialogue:**
  > "To automate ultrasound scanning, standard systems use open-loop raster or zigzag sweeps. However, human anatomy is curved and obstructed by bones like ribs. A blind sweep repeatedly scans flat skin while leaving structures behind bone in permanent dark acoustic shadow.  
  > We formulate active exploration using a submodular information-gain objective: re-scanning covered tissue yields diminishing returns, encouraging the robot to tilt the probe around acoustic obstacles.  
  > To be completely transparent with the committee: this formulation is implemented in code inside our simulator; conducting systematic empirical tests to prove whether it outperforms standard raster sweeps in scan time and volume completeness is our scheduled next step."
* **Panel Questions Cleared in Dialogue:**
  * *Why not just do a simple zigzag sweep?* Blind sweeps waste time re-scanning flat areas and leave occluded tissue behind bone in permanent shadow.
  * *Have you proven active planning is better?* It is mathematically formulated in code; empirical comparison against raster sweeps is scheduled for the next phase.

---

## Slide 10: Technical Understanding: Three Core Decisions
* **Slide Title:** Technical Understanding: Three Core Decisions
* **Spoken Dialogue:**
  > "This slide summarizes the three core architectural decisions of our first two months:  
  > 1. **Network Transport:** Rather than ROS2, which adds multi-node daemons on clinical PCs, we chose **OpenIGTLink TCP** because it is the native protocol of 3D Slicer, providing direct binary streaming with zero middleware bloat.  
  > 2. **Ultrasound Generation:** Rather than 2D GANs that can hallucinate tissue boundaries and require retraining for each patient, we chose **GPU Ray Physics**, which deterministically models wave decay zero-shot on any patient CT scan.  
  > 3. **Planning Strategy:** Rather than open-loop heuristic rasters, we chose **Submodular Next-Best-View planning** to actively steer probe angles around bone shadows."
* **Panel Questions Cleared in Dialogue:**
  * *Why were these specific technical choices made?* Grounded in clinical compatibility, zero-shot generalizability, and obstacle-aware coverage.

---

## Slide 11: Future Direction (1/2): Testing & Interactive Control
* **Slide Title:** Future Direction (1/2): Testing & Interactive Control
* **Spoken Dialogue:**
  > "Looking ahead to our next project phase: our immediate priority is **Simulation Pipeline Testing & Verification**. We will stress-test the physics loop under rapid contact force transitions, profile GPU reslicing speeds across diverse CT scan sizes, and confirm network socket stability during prolonged sessions.  
  > In addition, we will connect 3D Slicer interactive markups so clinicians can delineate target regions directly on the 3D viewport, allowing our inverse kinematics solver to automatically resolve initial collision-free poses."
* **Panel Questions Cleared in Dialogue:**
  * *What is your immediate next step?* Rigorous pipeline testing and profiling before running high-level AI experiments.
  * *How will clinicians interact with the system?* By placing 3D interactive markups in 3D Slicer to define target scanning regions.

---

## Slide 12: Future Direction (2/2): Policy Training & AI Benchmarking
* **Slide Title:** Future Direction (2/2): Policy Training & AI Benchmarking
* **Spoken Dialogue:**
  > "In the second half of our roadmap, we focus on policy learning: we will train Continuous Flow Matching and diffusion policies to generate smooth 6-DoF probe velocities in simulation, using our submodular coverage rewards.  
  > We will then conduct systematic comparative benchmarks: measuring whether our active view planning successfully reduces acoustic shadows and scan time compared to standard raster sweeps. All benchmarks will be conducted rigorously in simulation, establishing an open testbed for multimodal Vision-Language-Action models."
* **Panel Questions Cleared in Dialogue:**
  * *Are you testing on real physical hardware?* All learning benchmarks are conducted in simulation; sim-to-real transfer is the target of the framework design.
  * *How will you measure AI performance?* Volume completeness percentage, acoustic shadow reduction, and total scan duration.

---

## Slide 13: Conclusion & Discussion
* **Slide Title:** Conclusion & Discussion
* **Spoken Dialogue:**
  > "To conclude: over the first two months, we developed the entire simulation platform and 3D reconstruction pipeline. The Slicer-PyBullet bridge is stable, the GPU ultrasound physics are grounded in wave mechanics, and the active view selection formulation is ready for testing.  
  > With this foundation in place, we are on track for systematic pipeline testing and AI policy benchmarking. Thank you for your time, and I look forward to your questions and feedback."
* **Panel Questions Cleared in Dialogue:**
  * *Did you achieve your 2-month milestone?* Yes, the foundational simulation bridge and physical ultrasound reconstruction pipeline are fully developed.

---

## Quick Reference: Top 5 Committee Grilling Points & 10-Second Answers

1. **"Why PyBullet instead of IsaacLab like SonoGym?"**  
   *Answer:* *"SonoGym used IsaacLab for GPU parallel RL, but explicitly assumed the probe stays glued to the skin, bypassing normal contact force physics. We use PyBullet at 1000 Hz to simulate continuous soft-tissue contact dynamics and keep-out safety barriers."*

2. **"Why not use ROS2 to connect to 3D Slicer?"**  
   *Answer:* *"3D Slicer has native OpenIGTLink support. Using direct binary TCP sockets avoids multi-node ROS daemon setup and IPC serialization latency on clinical workstations."*

3. **"Is your ultrasound generation just slicing CT images?"**  
   *Answer:* *"No. Slicing only gives raw Hounsfield CT numbers. Our GPU shader models exponential sound wave decay with depth, acoustic impedance boundary reflections, and Rayleigh speckle noise, which generates authentic ultrasound physics in under 2 ms."*

4. **"Have you proven active planning is better than raster sweeps?"**  
   *Answer:* *"No, we have mathematically formulated and implemented the active planner in simulation. Proving its performance against raster sweeps in scan time and volume completeness is the primary objective of our upcoming testing phase."*

5. **"What if rendering in 3D Slicer lags?"**  
   *Answer:* *"Our multi-rate clock decoupling ensures PyBullet physics continues stepping at 1000 Hz regardless of visual rendering frame rate. Even if the GUI stutters, physical contact stability is never compromised."*
