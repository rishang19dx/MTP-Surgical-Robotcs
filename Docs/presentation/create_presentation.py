#!/usr/bin/env python3
"""
create_presentation.py
Generates the MTP Mid-Term Evaluation PowerPoint presentation (CS399P)
for Rishang Yadav (Roll No: B23173), guided by Dr. Deepak Raina.
"""

import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return RGBColor(*(int(hex_str[i:i+2], 16) for i in (0, 2, 4)))

# Brand Color Palette
CLR_BG = hex_to_rgb('F8FAFC')         # Light clean slate background
CLR_CARD_BG = hex_to_rgb('FFFFFF')    # Pure white card background
CLR_CARD_BORDER = hex_to_rgb('E2E8F0')# Light border
CLR_PRIMARY = hex_to_rgb('8C1D22')    # IIT Mandi Maroon
CLR_DARK = hex_to_rgb('0F172A')       # Dark Charcoal / Slate
CLR_TEXT_PRI = hex_to_rgb('1E293B')   # Primary text
CLR_TEXT_SEC = hex_to_rgb('475569')   # Muted body text
CLR_ACCENT_BLUE = hex_to_rgb('0284C7')# Medical Cyan/Blue
CLR_ACCENT_TEAL = hex_to_rgb('0D9488')# Deep Teal
CLR_ACCENT_AMBER = hex_to_rgb('D97706')# Warm Amber
CLR_PILL_BG = hex_to_rgb('F1F5F9')    # Pill grey-blue
CLR_SHADOW_BOX = hex_to_rgb('F1F5F9') # Soft grey box
CLR_HIGHLIGHT_BG = hex_to_rgb('FEF2F2') # Soft maroon tint
CLR_HIGHLIGHT_BORDER = hex_to_rgb('F87171') # Soft maroon border

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # completely blank layout

    def set_slide_background(slide, color=CLR_BG):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = color
        bg.line.color.rgb = color
        # send to back via python-pptx shape order if needed, but it's first so it is back.
        return bg

    def add_header(slide, category_text, title_text, subtitle_text=""):
        # Category Pill
        pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.4), Inches(2.8), Inches(0.35))
        pill.fill.solid()
        pill.fill.fore_color.rgb = hex_to_rgb('FEE2E2')
        pill.line.color.rgb = CLR_PRIMARY
        pill.line.width = Pt(1)
        tf_pill = pill.text_frame
        tf_pill.word_wrap = True
        tf_pill.margin_left = tf_pill.margin_right = tf_pill.margin_top = tf_pill.margin_bottom = 0
        p_pill = tf_pill.paragraphs[0]
        p_pill.text = category_text.upper()
        p_pill.font.size = Pt(10)
        p_pill.font.bold = True
        p_pill.font.color.rgb = CLR_PRIMARY
        p_pill.alignment = PP_ALIGN.CENTER

        # Slide Title
        tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.8), Inches(11.7), Inches(0.6))
        tf_title = tb_title.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_right = tf_title.margin_top = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = CLR_DARK

        if subtitle_text:
            p_sub = tf_title.add_paragraph()
            p_sub.text = subtitle_text
            p_sub.font.size = Pt(12)
            p_sub.font.color.rgb = CLR_TEXT_SEC
            p_sub.font.bold = False

    def add_card(slide, left, top, width, height, bg_color=CLR_CARD_BG, border_color=CLR_CARD_BORDER, border_width=1):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        if border_color:
            card.line.color.rgb = border_color
            card.line.width = Pt(border_width)
        else:
            card.line.fill.background()
        return card

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide1, CLR_BG)

    # Top accent bar
    top_bar = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.15))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = CLR_PRIMARY
    top_bar.line.fill.background()

    # Main Title Card
    main_card = add_card(slide1, Inches(0.8), Inches(0.8), Inches(11.733), Inches(5.9), CLR_CARD_BG, CLR_CARD_BORDER)

    # Title text box
    tb1 = slide1.shapes.add_textbox(Inches(1.2), Inches(1.1), Inches(11.0), Inches(2.2))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "CS399P: Major Technical Project | Mid-Term Evaluation"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = CLR_PRIMARY

    p2 = tf1.add_paragraph()
    p2.text = "Sim-to-Real Medical Robotics:"
    p2.font.size = Pt(30)
    p2.font.bold = True
    p2.font.color.rgb = CLR_DARK
    p2.space_before = Pt(8)

    p3 = tf1.add_paragraph()
    p3.text = "Interactive 3D Ultrasound Simulation & Learning Platform"
    p3.font.size = Pt(22)
    p3.font.bold = True
    p3.font.color.rgb = CLR_ACCENT_BLUE

    p4 = tf1.add_paragraph()
    p4.text = "A physics-grounded, hospital-grade simulation environment for autonomous ultrasound scanning and VLA benchmarking"
    p4.font.size = Pt(13)
    p4.font.color.rgb = CLR_TEXT_SEC
    p4.space_before = Pt(6)

    # Two Objective Badges in Slide 1
    badge1 = add_card(slide1, Inches(1.2), Inches(3.6), Inches(5.3), Inches(1.2), CLR_SHADOW_BOX, CLR_CARD_BORDER)
    tb_b1 = slide1.shapes.add_textbox(Inches(1.4), Inches(3.7), Inches(4.9), Inches(1.0))
    tf_b1 = tb_b1.text_frame
    tf_b1.word_wrap = True
    pb1_1 = tf_b1.paragraphs[0]
    pb1_1.text = "OBJECTIVE 1: SIMULATION PLATFORM"
    pb1_1.font.size = Pt(11)
    pb1_1.font.bold = True
    pb1_1.font.color.rgb = CLR_PRIMARY
    pb1_2 = tf_b1.add_paragraph()
    pb1_2.text = "PyBullet 1000 Hz dynamics coupled with 3D Slicer via OpenIGTLink, GPU CT acoustic reslicing, and virtual safety fixtures."
    pb1_2.font.size = Pt(11)
    pb1_2.font.color.rgb = CLR_TEXT_SEC

    badge2 = add_card(slide1, Inches(6.8), Inches(3.6), Inches(5.3), Inches(1.2), CLR_SHADOW_BOX, CLR_CARD_BORDER)
    tb_b2 = slide1.shapes.add_textbox(Inches(7.0), Inches(3.7), Inches(4.9), Inches(1.0))
    tf_b2 = tb_b2.text_frame
    tf_b2.word_wrap = True
    pb2_1 = tf_b2.paragraphs[0]
    pb2_1.text = "OBJECTIVE 2: LEARNING & BENCHMARKING"
    pb2_1.font.size = Pt(11)
    pb2_1.font.bold = True
    pb2_1.font.color.rgb = CLR_ACCENT_BLUE
    pb2_2 = tf_b2.add_paragraph()
    pb2_2.text = "Active Next-Best-View planning, multimodal VLA & diffusion policies, and sim-to-real transfer with tissue-mimicking phantoms."
    pb2_2.font.size = Pt(11)
    pb2_2.font.color.rgb = CLR_TEXT_SEC

    # Metadata Footer (Student & Supervisor)
    meta_box = slide1.shapes.add_textbox(Inches(1.2), Inches(5.2), Inches(10.9), Inches(1.2))
    tf_m = meta_box.text_frame
    tf_m.word_wrap = True
    pm1 = tf_m.paragraphs[0]
    pm1.text = "Candidate: Rishang Yadav (Roll No: B23173)   |   Supervisor: Dr. Deepak Raina"
    pm1.font.size = Pt(13)
    pm1.font.bold = True
    pm1.font.color.rgb = CLR_DARK

    pm2 = tf_m.add_paragraph()
    pm2.text = "School of Computing and Electrical Engineering (SCEE), Indian Institute of Technology Mandi"
    pm2.font.size = Pt(11)
    pm2.font.color.rgb = CLR_TEXT_SEC

    slide1.notes_slide.notes_text_frame.text = (
        "Good morning respected evaluation committee members and Dr. Raina. "
        "Welcome to the mid-term evaluation of my Major Technical Project, course CS399P, titled "
        "'Sim-to-Real Medical Robotics: Interactive 3D Ultrasound Simulation & Learning Platform'.\n\n"
        "Project Scope: This project tackles a foundational challenge in modern surgical robotics: robots operating in "
        "complex, uncertain, contact-rich environments require vast amounts of high-quality data to generalize. "
        "Nowhere is this more pressing than medical ultrasound, where probe contact, tissue pressure, and acoustic image "
        "formation are tightly coupled to physical dynamics.\n\n"
        "Our work addresses this via two tightly coupled goals: First, engineering a high-performance simulation platform "
        "bridging PyBullet physics to clinical 3D Slicer via OpenIGTLink with GPU acoustic ray tracing. Second, establishing "
        "an autonomous learning and benchmarking testbed for Next-Best-View reconstruction and VLA policies."
    )

    # =========================================================================
    # SLIDE 2: PROBLEM & BACKGROUND: CLINICAL MOTIVATION
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide2)
    add_header(slide2, "1. Problem & Background", "Clinical Motivation: The Robotic Ultrasound Dilemma",
               "Why autonomous ultrasound requires a physics-grounded simulation foundation")

    # 3 Column Cards
    col_width = Inches(3.64)
    col_gap = Inches(0.4)
    start_left = Inches(0.8)
    card_top = Inches(1.7)
    card_h = Inches(4.3)

    # Card 1: Operator Dependency
    add_card(slide2, start_left, card_top, col_width, card_h)
    tb_c1 = slide2.shapes.add_textbox(start_left + Inches(0.2), card_top + Inches(0.2), col_width - Inches(0.4), card_h - Inches(0.4))
    tf_c1 = tb_c1.text_frame
    tf_c1.word_wrap = True
    p = tf_c1.paragraphs[0]
    p.text = "OPERATOR DEPENDENT"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = CLR_PRIMARY

    p = tf_c1.add_paragraph()
    p.text = "High Cognitive & Physical Load"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = CLR_DARK
    p.space_before = Pt(4)

    p = tf_c1.add_paragraph()
    p.text = "\n• Sonography requires subtle, continuous adjustments of 6-DoF probe orientation, tilt, and acoustic gel coupling."
    p.font.size = Pt(11)
    p.font.color.rgb = CLR_TEXT_SEC

    p = tf_c1.add_paragraph()
    p.text = "\n• Over 80% of sonographers report work-related musculoskeletal disorders (WRMSD) due to sustained contact force application."
    p.font.size = Pt(11)
    p.font.color.rgb = CLR_TEXT_SEC

    p = tf_c1.add_paragraph()
    p.text = "\n• Diagnostic accuracy varies significantly with practitioner experience and fatigue."
    p.font.size = Pt(11)
    p.font.color.rgb = CLR_TEXT_SEC

    # Card 2: Contact-Rich Dynamics
    add_card(slide2, start_left + col_width + col_gap, card_top, col_width, card_h)
    tb_c2 = slide2.shapes.add_textbox(start_left + col_width + col_gap + Inches(0.2), card_top + Inches(0.2), col_width - Inches(0.4), card_h - Inches(0.4))
    tf_c2 = tb_c2.text_frame
    tf_c2.word_wrap = True
    p = tf_c2.paragraphs[0]
    p.text = "COUPLED DYNAMICS"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = CLR_ACCENT_BLUE

    p = tf_c2.add_paragraph()
    p.text = "Perception Coupled to Force"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = CLR_DARK
    p.space_before = Pt(4)

    p = tf_c2.add_paragraph()
    p.text = "\n• Unlike optical camera robotics, an ultrasound probe sees NOTHING without continuous physical contact with skin."
    p.font.size = Pt(11)
    p.font.color.rgb = CLR_TEXT_SEC

    p = tf_c2.add_paragraph()
    p.text = "\n• Too little force: acoustic acoustic dropouts and massive shadow artifacts."
    p.font.size = Pt(11)
    p.font.color.rgb = CLR_TEXT_SEC

    p = tf_c2.add_paragraph()
    p.text = "\n• Too much force: tissue deformation, patient discomfort, and internal organ displacement."
    p.font.size = Pt(11)
    p.font.color.rgb = CLR_TEXT_SEC

    # Card 3: Data Scarcity Bottleneck
    add_card(slide2, start_left + (col_width + col_gap)*2, card_top, col_width, card_h)
    tb_c3 = slide2.shapes.add_textbox(start_left + (col_width + col_gap)*2 + Inches(0.2), card_top + Inches(0.2), col_width - Inches(0.4), card_h - Inches(0.4))
    tf_c3 = tb_c3.text_frame
    tf_c3.word_wrap = True
    p = tf_c3.paragraphs[0]
    p.text = "DATA SCARCITY"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = CLR_ACCENT_AMBER

    p = tf_c3.add_paragraph()
    p.text = "Sim-to-Real is Mandatory"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = CLR_DARK
    p.space_before = Pt(4)

    p = tf_c3.add_paragraph()
    p.text = "\n• Reinforcement learning or trial-and-error training directly on human patients is ethically and practically impossible."
    p.font.size = Pt(11)
    p.font.color.rgb = CLR_TEXT_SEC

    p = tf_c3.add_paragraph()
    p.text = "\n• Real-world robot teleoperation on physical silicone phantoms is slow, costly, and lacks anatomical diversity."
    p.font.size = Pt(11)
    p.font.color.rgb = CLR_TEXT_SEC

    p = tf_c3.add_paragraph()
    p.text = "\n• A physics-faithful simulator is the ONLY viable pipeline to generate millions of paired multimodal training sweeps."
    p.font.size = Pt(11)
    p.font.color.rgb = CLR_TEXT_SEC

    # Bottom Takeaway Strip
    strip = add_card(slide2, Inches(0.8), Inches(6.2), Inches(11.733), Inches(0.8), CLR_SHADOW_BOX, CLR_CARD_BORDER)
    tb_strip = slide2.shapes.add_textbox(Inches(1.0), Inches(6.25), Inches(11.3), Inches(0.7))
    tf_strip = tb_strip.text_frame
    tf_strip.word_wrap = True
    p_s = tf_strip.paragraphs[0]
    p_s.text = "Core Takeaway: To train AI policies for healthcare, simulation must model contact forces, soft-tissue compliance, and acoustic wave formation simultaneously."
    p_s.font.size = Pt(12)
    p_s.font.bold = True
    p_s.font.color.rgb = CLR_DARK

    slide2.notes_slide.notes_text_frame.text = (
        "Slide 2 Speaker Notes:\n"
        "To motivate our project clearly: Why robotic ultrasound? Why can't we just collect data on physical robots like in standard warehouse manipulation?\n\n"
        "First, ultrasound is notoriously operator-dependent. A slight rotation or pressure shift completely alters image clarity. Sonographers experience severe musculoskeletal injuries from maintaining steady probe pressure for hours.\n\n"
        "Second, ultrasound represents an extreme case of contact-rich dynamics. In optical vision, if a robot pulls back 2 cm, it still sees the scene. In ultrasound, if contact drops by 1 mm, total acoustic acoustic reflection occurs and the image becomes complete black noise. Furthermore, excessive force compresses blood vessels and causes patient pain.\n\n"
        "Third, data scarcity: We cannot let an untrained neural network or RL agent explore policies on living patients. Real phantom scans are too slow to scale. Simulation is the indispensable prerequisite to generate the millions of synchronized multimodal trajectories needed to train robust medical AI models."
    )

    # =========================================================================
    # SLIDE 3: PROBLEM & BACKGROUND: PROJECT OBJECTIVES
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide3)
    add_header(slide3, "1. Problem & Background", "Project Objectives: Platform Development & AI Benchmarking",
               "Decomposing the MTP into two synergistic technical milestones")

    box_w = Inches(5.66)
    box_h = Inches(4.8)
    box_top = Inches(1.8)

    # Left Track Box: Simulation Platform
    add_card(slide3, Inches(0.8), box_top, box_w, box_h)
    tb_t1 = slide3.shapes.add_textbox(Inches(1.1), box_top + Inches(0.2), box_w - Inches(0.6), box_h - Inches(0.4))
    tf_t1 = tb_t1.text_frame
    tf_t1.word_wrap = True
    p = tf_t1.paragraphs[0]
    p.text = "OBJECTIVE 1"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = CLR_PRIMARY

    p = tf_t1.add_paragraph()
    p.text = "Simulation Platform Development"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = CLR_DARK
    p.space_before = Pt(4)

    p = tf_t1.add_paragraph()
    p.text = "Building a clinical-grade, physics-accurate simulation environment:"
    p.font.size = Pt(11)
    p.font.color.rgb = CLR_TEXT_SEC
    p.space_before = Pt(8)

    items_t1 = [
        ("High-Frequency Dynamics", "1000 Hz PyBullet physics simulating a Kinova Gen3 7-DoF arm with realistic joint friction, limits, and soft tissue contact."),
        ("Medical Suite Integration", "Direct OpenIGTLink binary link into 3D Slicer at 60 FPS without middleware or ROS overhead."),
        ("GPU Acoustic Wave Physics", "PyTorch GPU ray reslicing on patient CT scans modeling acoustic attenuation, impedance reflections, and speckle noise."),
        ("Virtual Safety Fixtures", "Enforcing keep-out safety zones and contact force boundaries to guarantee patient safety constraints.")
    ]
    for title, desc in items_t1:
        p = tf_t1.add_paragraph()
        p.text = f"\n• {title}: {desc}"
        p.font.size = Pt(11)
        p.font.color.rgb = CLR_TEXT_PRI

    # Right Track Box: Learning & Benchmarking
    add_card(slide3, Inches(6.866), box_top, box_w, box_h)
    tb_t2 = slide3.shapes.add_textbox(Inches(7.166), box_top + Inches(0.2), box_w - Inches(0.6), box_h - Inches(0.4))
    tf_t2 = tb_t2.text_frame
    tf_t2.word_wrap = True
    p = tf_t2.paragraphs[0]
    p.text = "OBJECTIVE 2"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = CLR_ACCENT_BLUE

    p = tf_t2.add_paragraph()
    p.text = "Learning & Benchmarking Suite"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = CLR_DARK
    p.space_before = Pt(4)

    p = tf_t2.add_paragraph()
    p.text = "Developing autonomous AI policies and benchmarking sim-to-real transfer:"
    p.font.size = Pt(11)
    p.font.color.rgb = CLR_TEXT_SEC
    p.space_before = Pt(8)

    items_t2 = [
        ("Active 3D Reconstruction", "Autonomous volumetric scanning via submodular Next-Best-View (NBV) planning to maximize volume coverage with minimal sweeps."),
        ("Multimodal VLA Policies", "Benchmarking Vision-Language-Action models and diffusion action-chunking conditioned on acoustic images, force, and text instructions."),
        ("Sim-to-Real Transferability", "Validating policy transfer from virtual patient CTs to physical silicone phantoms and robotic hardware."),
        ("Benchmarking Suite", "Establishing standardized metrics: scan duration, acoustic coverage, reconstruction fidelity, and contact force safety.")
    ]
    for title, desc in items_t2:
        p = tf_t2.add_paragraph()
        p.text = f"\n• {title}: {desc}"
        p.font.size = Pt(11)
        p.font.color.rgb = CLR_TEXT_PRI

    slide3.notes_slide.notes_text_frame.text = (
        "Slide 3 Speaker Notes:\n"
        "Here we formally divide the MTP into two distinct, interconnected tracks.\n\n"
        "Track 1 is Platform Development: Most existing ultrasound simulators are either crude game-engine demos or locked inside proprietary commercial frameworks. We require an open, high-performance platform capable of simulating 1000 Hz robot contact dynamics, coupled seamlessly with 3D Slicer—the gold standard in clinical imaging—via the OpenIGTLink medical robotics protocol, while simulating realistic ultrasound image formation on the GPU.\n\n"
        "Track 2 is Learning and Benchmarking: With this simulator in hand, we tackle the clinical task: Autonomous 3D ultrasound reconstruction. The robot must intelligently sweep the probe over patient anatomy, choosing Next-Best-Views to fill in acoustic shadows and uncertainty, laying the foundation for modern Vision-Language-Action (VLA) models and sim-to-real evaluation."
    )

    # =========================================================================
    # SLIDE 4: LITERATURE SURVEY & BASELINE LIMITATIONS
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide4)
    add_header(slide4, "2. Literature Survey", "Prior Work & Baseline Limitations",
               "Evaluating state-of-the-art simulators and AI frameworks to establish our technical differentiators")

    c_w = Inches(2.7)
    c_gap = Inches(0.3)
    c_top = Inches(1.8)
    c_h = Inches(4.8)

    # Baseline 1: SonoGym
    add_card(slide4, start_left, c_top, c_w, c_h)
    tb_l1 = slide4.shapes.add_textbox(start_left + Inches(0.15), c_top + Inches(0.15), c_w - Inches(0.3), c_h - Inches(0.3))
    tf_l1 = tb_l1.text_frame
    tf_l1.word_wrap = True
    p = tf_l1.paragraphs[0]
    p.text = "SonoGym (2023)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = CLR_DARK

    p = tf_l1.add_paragraph()
    p.text = "RL Gym for 2D Tracking"
    p.font.size = Pt(11)
    p.font.color.rgb = CLR_ACCENT_BLUE
    p.space_before = Pt(2)

    p = tf_l1.add_paragraph()
    p.text = "\nWhat They Did:\n• Fast PyBullet gym environment for training 2D probe tracking and visual servoing."
    p.font.size = Pt(10)
    p.font.color.rgb = CLR_TEXT_SEC

    p = tf_l1.add_paragraph()
    p.text = "\nKey Limitations:\n• Rigid static anatomy; no soft-tissue deformation.\n• Mock ultrasound slices fetched from fixed pre-rendered stacks without acoustic ray physics.\n• No volumetric 3D reconstruction."
    p.font.size = Pt(10)
    p.font.color.rgb = CLR_PRIMARY

    # Baseline 2: SRT / SRT-H
    add_card(slide4, start_left + c_w + c_gap, c_top, c_w, c_h)
    tb_l2 = slide4.shapes.add_textbox(start_left + c_w + c_gap + Inches(0.15), c_top + Inches(0.15), c_w - Inches(0.3), c_h - Inches(0.3))
    tf_l2 = tb_l2.text_frame
    tf_l2.word_wrap = True
    p = tf_l2.paragraphs[0]
    p.text = "SRT & SRT-H (2024)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = CLR_DARK

    p = tf_l2.add_paragraph()
    p.text = "Surgical VLA Transformers"
    p.font.size = Pt(11)
    p.font.color.rgb = CLR_ACCENT_BLUE
    p.space_before = Pt(2)

    p = tf_l2.add_paragraph()
    p.text = "\nWhat They Did:\n• State-of-the-art imitation learning with diffusion policies & hybrid action tokens for surgical robots."
    p.font.size = Pt(10)
    p.font.color.rgb = CLR_TEXT_SEC

    p = tf_l2.add_paragraph()
    p.text = "\nKey Limitations:\n• Evaluated purely on rigid peg-transfer and block manipulation.\n• Lacks acoustic imaging feedback and continuous contact force compliance."
    p.font.size = Pt(10)
    p.font.color.rgb = CLR_PRIMARY

    # Baseline 3: AMBF & SlicerROS2
    add_card(slide4, start_left + (c_w + c_gap)*2, c_top, c_w, c_h)
    tb_l3 = slide4.shapes.add_textbox(start_left + (c_w + c_gap)*2 + Inches(0.15), c_top + Inches(0.15), c_w - Inches(0.3), c_h - Inches(0.3))
    tf_l3 = tb_l3.text_frame
    tf_l3.word_wrap = True
    p = tf_l3.paragraphs[0]
    p.text = "AMBF & SlicerROS2"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = CLR_DARK

    p = tf_l3.add_paragraph()
    p.text = "ROS Middleware Bridges"
    p.font.size = Pt(11)
    p.font.color.rgb = CLR_ACCENT_BLUE
    p.space_before = Pt(2)

    p = tf_l3.add_paragraph()
    p.text = "\nWhat They Did:\n• ROS/ROS2-based frameworks linking surgical simulators to 3D Slicer for visualization."
    p.font.size = Pt(10)
    p.font.color.rgb = CLR_TEXT_SEC

    p = tf_l3.add_paragraph()
    p.text = "\nKey Limitations:\n• Heavy multi-process serialization overhead.\n• Significant IPC latency and jitter during high-frequency robot control.\n• Cumbersome multi-node setup prone to message drops."
    p.font.size = Pt(10)
    p.font.color.rgb = CLR_PRIMARY

    # Column 4: OUR DIFFERENTIATORS (Highlighted)
    add_card(slide4, start_left + (c_w + c_gap)*3, c_top, c_w, c_h, CLR_HIGHLIGHT_BG, CLR_HIGHLIGHT_BORDER, border_width=2)
    tb_l4 = slide4.shapes.add_textbox(start_left + (c_w + c_gap)*3 + Inches(0.15), c_top + Inches(0.15), c_w - Inches(0.3), c_h - Inches(0.3))
    tf_l4 = tb_l4.text_frame
    tf_l4.word_wrap = True
    p = tf_l4.paragraphs[0]
    p.text = "OUR APPROACH"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = CLR_PRIMARY

    p = tf_l4.add_paragraph()
    p.text = "Direct, Physically Grounded"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CLR_DARK
    p.space_before = Pt(2)

    p = tf_l4.add_paragraph()
    p.text = "\nWhat We Do Differently:\n• Direct OpenIGTLink binary link (Zero ROS overhead, sub-millisecond latency).\n• Live GPU CT reslicing + physical acoustic ray tracing (attenuation, reflection, speckle).\n• Soft-tissue compliance + real-time keep-out virtual safety fixtures.\n• Active 3D Next-Best-View volumetric reconstruction."
    p.font.size = Pt(10)
    p.font.color.rgb = CLR_DARK

    slide4.notes_slide.notes_text_frame.text = (
        "Slide 4 Speaker Notes:\n"
        "Let's position our contribution squarely against the literature.\n\n"
        "SonoGym was an important step for RL in ultrasound, but it used rigid anatomy and fake pre-rendered 2D slices. There was no true acoustic ray physics, no tissue deformation, and no 3D volumetric reconstruction.\n\n"
        "Surgical Robot Transformers (SRT & SRT-H) demonstrate the incredible power of diffusion action chunking and language-conditioned surgical policies, but they tested exclusively on rigid tasks like peg-transfer. They did not incorporate contact-coupled sensing like ultrasound.\n\n"
        "Existing Slicer-robot connectors like AMBF or SlicerROS2 rely on ROS middleware. Anyone who has run high-rate robot loops with ROS2 knows that serializing large point clouds and transform trees into ROS messages introduces severe latency and jitter.\n\n"
        "Our approach departs from these baselines in three critical ways: We eliminate ROS entirely using OpenIGTLink binary sockets directly into 3D Slicer; we synthesize realistic B-mode ultrasound via GPU ray-tracing directly on clinical CT volumes; and we couple the reconstruction to an active Next-Best-View planning pipeline."
    )

    # =========================================================================
    # SLIDE 5: SYSTEM ARCHITECTURE & END-TO-END WORKFLOW
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide5)
    add_header(slide5, "3. Work Accomplished", "System Architecture & End-to-End Workflow",
               "Closed-loop integration of high-rate robot dynamics, GPU acoustic synthesis, and medical frontend")

    # Workflow Container Box
    flow_box = add_card(slide5, Inches(0.8), Inches(1.8), Inches(11.733), Inches(3.2), CLR_CARD_BG, CLR_CARD_BORDER)

    # Row 1: Diagram Boxes (Left to Right)
    box_y1 = Inches(2.1)
    bw = Inches(2.1)
    bh = Inches(1.1)

    # Step 1: Patient CT Volume
    s1 = add_card(slide5, Inches(1.1), box_y1, bw, bh, CLR_SHADOW_BOX, CLR_CARD_BORDER)
    tb = slide5.shapes.add_textbox(Inches(1.15), box_y1 + Inches(0.1), bw - Inches(0.1), bh - Inches(0.2))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "Patient 3D CT"; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = CLR_DARK; p.alignment = PP_ALIGN.CENTER
    p = tf.add_paragraph(); p.text = "Volumetric tensor (Hounsfield units)"; p.font.size = Pt(9.5); p.font.color.rgb = CLR_TEXT_SEC; p.alignment = PP_ALIGN.CENTER

    # Arrow 1
    ar1 = slide5.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(3.3), box_y1 + Inches(0.35), Inches(0.4), Inches(0.3))
    ar1.fill.solid(); ar1.fill.fore_color.rgb = CLR_PRIMARY; ar1.line.fill.background()

    # Step 2: GPU Reslicer
    s2 = add_card(slide5, Inches(3.8), box_y1, bw, bh, CLR_SHADOW_BOX, CLR_CARD_BORDER)
    tb = slide5.shapes.add_textbox(Inches(3.85), box_y1 + Inches(0.1), bw - Inches(0.1), bh - Inches(0.2))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "GPU Acoustic Shader"; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = CLR_PRIMARY; p.alignment = PP_ALIGN.CENTER
    p = tf.add_paragraph(); p.text = "grid_sample + attenuation & speckle"; p.font.size = Pt(9.5); p.font.color.rgb = CLR_TEXT_SEC; p.alignment = PP_ALIGN.CENTER

    # Arrow 2
    ar2 = slide5.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(6.0), box_y1 + Inches(0.35), Inches(0.4), Inches(0.3))
    ar2.fill.solid(); ar2.fill.fore_color.rgb = CLR_PRIMARY; ar2.line.fill.background()

    # Step 3: Synthesized B-Mode
    s3 = add_card(slide5, Inches(6.5), box_y1, bw, bh, CLR_SHADOW_BOX, CLR_CARD_BORDER)
    tb = slide5.shapes.add_textbox(Inches(6.55), box_y1 + Inches(0.1), bw - Inches(0.1), bh - Inches(0.2))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "Synthesized B-Mode"; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = CLR_DARK; p.alignment = PP_ALIGN.CENTER
    p = tf.add_paragraph(); p.text = "2D frame + confidence map"; p.font.size = Pt(9.5); p.font.color.rgb = CLR_TEXT_SEC; p.alignment = PP_ALIGN.CENTER

    # Arrow 3
    ar3 = slide5.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(8.7), box_y1 + Inches(0.35), Inches(0.4), Inches(0.3))
    ar3.fill.solid(); ar3.fill.fore_color.rgb = CLR_PRIMARY; ar3.line.fill.background()

    # Step 4: 3D Voxel Accumulator
    s4 = add_card(slide5, Inches(9.2), box_y1, bw, bh, CLR_SHADOW_BOX, CLR_CARD_BORDER)
    tb = slide5.shapes.add_textbox(Inches(9.25), box_y1 + Inches(0.1), bw - Inches(0.1), bh - Inches(0.2))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "3D Reconstruction"; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = CLR_ACCENT_BLUE; p.alignment = PP_ALIGN.CENTER
    p = tf.add_paragraph(); p.text = "Voxel fusion + uncertainty tracking"; p.font.size = Pt(9.5); p.font.color.rgb = CLR_TEXT_SEC; p.alignment = PP_ALIGN.CENTER

    # Row 2 (Bottom Row of Flow Diagram): Closed Loop Physics & GUI
    box_y2 = Inches(3.6)

    # Robot Physics Box
    s_phys = add_card(slide5, Inches(2.2), box_y2, Inches(3.8), Inches(1.1), CLR_SHADOW_BOX, CLR_CARD_BORDER)
    tb_p = slide5.shapes.add_textbox(Inches(2.3), box_y2 + Inches(0.1), Inches(3.6), Inches(0.9))
    tf_p = tb_p.text_frame; tf_p.word_wrap = True
    p = tf_p.paragraphs[0]; p.text = "PyBullet Physics Engine (1000 Hz)"; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = CLR_PRIMARY; p.alignment = PP_ALIGN.CENTER
    p = tf_p.add_paragraph(); p.text = "Kinova 7-DoF + Contact Forces + Virtual Safety Fixtures"; p.font.size = Pt(9.5); p.font.color.rgb = CLR_TEXT_SEC; p.alignment = PP_ALIGN.CENTER

    # Bi-directional link between Physics and Slicer
    link_shape = slide5.shapes.add_shape(MSO_SHAPE.LEFT_RIGHT_ARROW, Inches(6.1), box_y2 + Inches(0.35), Inches(0.8), Inches(0.35))
    link_shape.fill.solid(); link_shape.fill.fore_color.rgb = CLR_ACCENT_TEAL; link_shape.line.fill.background()

    # Slicer Box
    s_slic = add_card(slide5, Inches(7.0), box_y2, Inches(4.3), Inches(1.1), CLR_SHADOW_BOX, CLR_CARD_BORDER)
    tb_s = slide5.shapes.add_textbox(Inches(7.1), box_y2 + Inches(0.1), Inches(4.1), Inches(0.9))
    tf_s = tb_s.text_frame; tf_s.word_wrap = True
    p = tf_s.paragraphs[0]; p.text = "3D Slicer Clinical Suite (60 FPS)"; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = CLR_ACCENT_TEAL; p.alignment = PP_ALIGN.CENTER
    p = tf_s.add_paragraph(); p.text = "OpenIGTLink Protocol | In-Place VTK Updates | Doctor View"; p.font.size = Pt(9.5); p.font.color.rgb = CLR_TEXT_SEC; p.alignment = PP_ALIGN.CENTER

    # Small Write-up: 3 Decoupled Loops
    card_w3 = Inches(3.64)
    start_y3 = Inches(5.2)
    card_h3 = Inches(1.8)

    # Loop 1
    add_card(slide5, start_left, start_y3, card_w3, card_h3)
    tb_l1 = slide5.shapes.add_textbox(start_left + Inches(0.15), start_y3 + Inches(0.1), card_w3 - Inches(0.3), card_h3 - Inches(0.2))
    tf_l1 = tb_l1.text_frame; tf_l1.word_wrap = True
    p = tf_l1.paragraphs[0]; p.text = "High-Rate Physics Loop (1000 Hz)"; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = CLR_PRIMARY
    p = tf_l1.add_paragraph(); p.text = "Simulates rigid robot dynamics, tissue deformation, joint limits, and normal force responses at 1 ms intervals to prevent numerical contact instability."; p.font.size = Pt(10.5); p.font.color.rgb = CLR_TEXT_SEC; p.space_before = Pt(4)

    # Loop 2
    add_card(slide5, start_left + card_w3 + col_gap, start_y3, card_w3, card_h3)
    tb_l2 = slide5.shapes.add_textbox(start_left + card_w3 + col_gap + Inches(0.15), start_y3 + Inches(0.1), card_w3 - Inches(0.3), card_h3 - Inches(0.2))
    tf_l2 = tb_l2.text_frame; tf_l2.word_wrap = True
    p = tf_l2.paragraphs[0]; p.text = "Visual & Telemetry Loop (60 Hz)"; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = CLR_ACCENT_BLUE
    p = tf_l2.add_paragraph(); p.text = "Serializes robot kinematics and contact force vectors over direct OpenIGTLink TCP sockets to 3D Slicer, updating VTK mesh transforms smoothly."; p.font.size = Pt(10.5); p.font.color.rgb = CLR_TEXT_SEC; p.space_before = Pt(4)

    # Loop 3
    add_card(slide5, start_left + (card_w3 + col_gap)*2, start_y3, card_w3, card_h3)
    tb_l3 = slide5.shapes.add_textbox(start_left + (card_w3 + col_gap)*2 + Inches(0.15), start_y3 + Inches(0.1), card_w3 - Inches(0.3), card_h3 - Inches(0.2))
    tf_l3 = tb_l3.text_frame; tf_l3.word_wrap = True
    p = tf_l3.paragraphs[0]; p.text = "Reconstruction & Planning (10 Hz)"; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = CLR_ACCENT_AMBER
    p = tf_l3.add_paragraph(); p.text = "Accumulates synthesized 2D B-scans into a global 3D voxel grid and computes information gain to command the next-best probe trajectory."; p.font.size = Pt(10.5); p.font.color.rgb = CLR_TEXT_SEC; p.space_before = Pt(4)

    slide5.notes_slide.notes_text_frame.text = (
        "Slide 5 Speaker Notes:\n"
        "This slide presents the end-to-end system architecture of our platform.\n\n"
        "Notice how the data flows seamlessly across three decoupled tiers:\n"
        "At the top, we ingest patient volumetric CT scans. When the robot moves the probe in PyBullet, the exact 6-DoF probe transform is passed to our PyTorch GPU acoustic shader, which synthesizes the 2D B-mode ultrasound frame with realistic attenuation and speckle.\n\n"
        "In the middle, the synthesized B-scans feed into a 3D voxel reconstruction grid, accumulating scans and estimating regional scanning uncertainty.\n\n"
        "At the bottom is our multi-rate execution model: Robot physics must run at 1000 Hz to guarantee contact stability without jitter or penetration blow-ups. However, medical rendering in 3D Slicer only requires 60 FPS, and trajectory planning runs at 10 Hz. By decoupling these three loops across separate asynchronous threads, we maintain clinical visualization and AI planning without degrading real-time physical simulation."
    )

    # =========================================================================
    # SLIDE 6: WORK ACCOMPLISHED: SLICER SURGICAL BRIDGE
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide6)
    add_header(slide6, "3. Work Accomplished", "Slicer-PyBullet Surgical Bridge",
               "High-performance direct binary coupling between physics simulation and clinical visualization")

    # Flow Diagram on Top Half
    diag_box6 = add_card(slide6, Inches(0.8), Inches(1.8), Inches(11.733), Inches(2.2), CLR_CARD_BG, CLR_CARD_BORDER)

    # Diagram Elements
    dw = Inches(3.3)
    dh = Inches(1.6)
    dy = Inches(2.1)

    # Box 1: PyBullet Physics
    add_card(slide6, Inches(1.1), dy, dw, dh, CLR_SHADOW_BOX, CLR_CARD_BORDER)
    tb = slide6.shapes.add_textbox(Inches(1.2), dy + Inches(0.1), dw - Inches(0.2), dh - Inches(0.2))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "PyBullet Physics Engine"; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = CLR_PRIMARY
    p = tf.add_paragraph(); p.text = "• Kinova Gen3 7-DoF Robot Dynamics\n• 1000 Hz Contact Physics & Constraints\n• Keep-Out Virtual Safety Fixtures"; p.font.size = Pt(10.5); p.font.color.rgb = CLR_TEXT_PRI; p.space_before = Pt(4)

    # Arrow with label
    ar6 = slide6.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(4.6), dy + Inches(0.5), Inches(1.0), Inches(0.4))
    ar6.fill.solid(); ar6.fill.fore_color.rgb = CLR_ACCENT_TEAL; ar6.line.fill.background()
    tb_ar = slide6.shapes.add_textbox(Inches(4.5), dy + Inches(0.1), Inches(1.2), Inches(0.4))
    tf_ar = tb_ar.text_frame; tf_ar.word_wrap = True
    p = tf_ar.paragraphs[0]; p.text = "TCP Sockets"; p.font.size = Pt(9.5); p.font.bold = True; p.font.color.rgb = CLR_ACCENT_TEAL; p.alignment = PP_ALIGN.CENTER

    # Box 2: OpenIGTLink Protocol
    add_card(slide6, Inches(5.8), dy, Inches(2.3), dh, CLR_SHADOW_BOX, CLR_CARD_BORDER)
    tb = slide6.shapes.add_textbox(Inches(5.85), dy + Inches(0.1), Inches(2.2), dh - Inches(0.2))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "OpenIGTLink"; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = CLR_DARK; p.alignment = PP_ALIGN.CENTER
    p = tf.add_paragraph(); p.text = "Standard Medical Robotics Binary Transport (Header + Body CRC)"; p.font.size = Pt(10); p.font.color.rgb = CLR_TEXT_SEC; p.alignment = PP_ALIGN.CENTER; p.space_before = Pt(4)

    # Arrow 2
    ar6_2 = slide6.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(8.3), dy + Inches(0.5), Inches(0.8), Inches(0.4))
    ar6_2.fill.solid(); ar6_2.fill.fore_color.rgb = CLR_ACCENT_TEAL; ar6_2.line.fill.background()

    # Box 3: 3D Slicer Frontend
    add_card(slide6, Inches(9.3), dy, dw, dh, CLR_SHADOW_BOX, CLR_CARD_BORDER)
    tb = slide6.shapes.add_textbox(Inches(9.4), dy + Inches(0.1), dw - Inches(0.2), dh - Inches(0.2))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "3D Slicer Clinical Suite"; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = CLR_ACCENT_TEAL
    p = tf.add_paragraph(); p.text = "• 60 FPS In-Place VTK Mesh Transforms\n• Live Dynamic Force Vector Widget\n• Native Surgeon Planning UI"; p.font.size = Pt(10.5); p.font.color.rgb = CLR_TEXT_PRI; p.space_before = Pt(4)

    # Write-up: 3 Small Cards (Bottom Half)
    b_top = Inches(4.3)
    b_h = Inches(2.7)

    # Card 1: Zero-ROS OpenIGTLink
    add_card(slide6, start_left, b_top, card_w3, b_h)
    tb_b1 = slide6.shapes.add_textbox(start_left + Inches(0.15), b_top + Inches(0.15), card_w3 - Inches(0.3), b_h - Inches(0.3))
    tf_b1 = tb_b1.text_frame; tf_b1.word_wrap = True
    p = tf_b1.paragraphs[0]; p.text = "Zero-ROS Architecture"; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = CLR_PRIMARY
    p = tf_b1.add_paragraph(); p.text = "\n• Why not ROS/ROS2?\nROS adds heavy serialization, daemon processes, and message drop issues on non-Linux surgical workstations."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_SEC
    p = tf_b1.add_paragraph(); p.text = "\n• Direct OpenIGTLink:\nUses standard medical robotics protocol over lightweight TCP binary sockets with sub-millisecond round-trip latency."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_SEC

    # Card 2: 1000 Hz Physics vs 60 FPS Render
    add_card(slide6, start_left + card_w3 + col_gap, b_top, card_w3, b_h)
    tb_b2 = slide6.shapes.add_textbox(start_left + card_w3 + col_gap + Inches(0.15), b_top + Inches(0.15), card_w3 - Inches(0.3), b_h - Inches(0.3))
    tf_b2 = tb_b2.text_frame; tf_b2.word_wrap = True
    p = tf_b2.paragraphs[0]; p.text = "Multi-Rate Rate Decoupling"; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = CLR_ACCENT_BLUE
    p = tf_b2.add_paragraph(); p.text = "\n• Stable Contact Physics:\nRigid dynamics and penalty-based contact constraints resolve at 1000 Hz (1 ms timestep), preventing probe jitter."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_SEC
    p = tf_b2.add_paragraph(); p.text = "\n• In-Place VTK Updates:\n3D Slicer receives state transforms and updates existing scene graph nodes in-place, eliminating memory leaks."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_SEC

    # Card 3: Virtual Safety Fixtures
    add_card(slide6, start_left + (card_w3 + col_gap)*2, b_top, card_w3, b_h)
    tb_b3 = slide6.shapes.add_textbox(start_left + (card_w3 + col_gap)*2 + Inches(0.15), b_top + Inches(0.15), card_w3 - Inches(0.3), b_h - Inches(0.3))
    tf_b3 = tb_b3.text_frame; tf_b3.word_wrap = True
    p = tf_b3.paragraphs[0]; p.text = "Virtual Safety Fixtures"; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = CLR_ACCENT_AMBER
    p = tf_b3.add_paragraph(); p.text = "\n• Task-Space Keep-Out Zones:\nSurgeon defines critical organ boundaries (e.g. carotid artery) in 3D Slicer; bridge streams JSON safety geometry."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_SEC
    p = tf_b3.add_paragraph(); p.text = "\n• Active Force Projection:\nKinova controller projects velocity into the nullspace of boundary normals, making penetration physically impossible."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_SEC

    slide6.notes_slide.notes_text_frame.text = (
        "Slide 6 Speaker Notes:\n"
        "This slide covers our first major accomplished milestone: The Slicer-PyBullet Surgical Bridge.\n\n"
        "Let's look at the flow diagram on top: We have PyBullet executing physics at 1000 Hz, packaging joint angles and contact force vectors into OpenIGTLink binary frames, and sending them directly over a TCP socket to 3D Slicer, which updates VTK mesh nodes at 60 FPS.\n\n"
        "Why did we make this design choice instead of standard ROS2?\n"
        "1. ROS2 introduces massive serialization and IPC overhead. Furthermore, hospitals and surgeons do not run Ubuntu with ROS; they run 3D Slicer on clinical workstations. OpenIGTLink is the established international standard protocol for image-guided robotic surgery.\n"
        "2. Rate Decoupling: Contact mechanics in physics require 1000 Hz so the probe doesn't bounce or tunnel through the body. But 3D rendering cannot and should not run at 1000 Hz. Our socket bridge downsamples cleanly to 60 FPS while keeping the physics rock solid.\n"
        "3. Virtual Fixtures: We implemented real-time keep-out safety zones. If a physician marks a critical blood vessel in 3D Slicer, the boundary is sent to PyBullet, which constrains the robot's Jacobian controller to prevent the probe from ever violating that zone."
    )

    # =========================================================================
    # SLIDE 7: WORK ACCOMPLISHED: 3D ULTRASOUND RECONSTRUCTION PIPELINE
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide7)
    add_header(slide7, "3. Work Accomplished", "3D Ultrasound Reconstruction Pipeline",
               "GPU-accelerated acoustic physics reslicing, confidence accumulation, and active view planning")

    # Flow Diagram on Top Half
    diag_box7 = add_card(slide7, Inches(0.8), Inches(1.8), Inches(11.733), Inches(2.2), CLR_CARD_BG, CLR_CARD_BORDER)

    # 4 Flow Steps in Diagram
    dw7 = Inches(2.3)
    dh7 = Inches(1.6)
    dy7 = Inches(2.1)
    gap7 = Inches(0.6)

    # Step 1: 3D CT Volume
    add_card(slide7, Inches(1.1), dy7, dw7, dh7, CLR_SHADOW_BOX, CLR_CARD_BORDER)
    tb = slide7.shapes.add_textbox(Inches(1.15), dy7 + Inches(0.1), dw7 - Inches(0.1), dh7 - Inches(0.2))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "1. Patient CT Volume"; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = CLR_DARK
    p = tf.add_paragraph(); p.text = "• 3D density tensor\n• Hounsfield Units\n• Ground-truth anatomy"; p.font.size = Pt(10); p.font.color.rgb = CLR_TEXT_SEC; p.space_before = Pt(4)

    # Arrow 1
    ar = slide7.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(3.45), dy7 + Inches(0.6), Inches(0.45), Inches(0.3))
    ar.fill.solid(); ar.fill.fore_color.rgb = CLR_PRIMARY; ar.line.fill.background()

    # Step 2: GPU Acoustic Slicing
    add_card(slide7, Inches(4.0), dy7, dw7, dh7, CLR_SHADOW_BOX, CLR_CARD_BORDER)
    tb = slide7.shapes.add_textbox(Inches(4.05), dy7 + Inches(0.1), dw7 - Inches(0.1), dh7 - Inches(0.2))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "2. Acoustic GPU Shader"; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = CLR_PRIMARY
    p = tf.add_paragraph(); p.text = "• grid_sample reslicing\n• Depth attenuation\n• Acoustic speckle noise"; p.font.size = Pt(10); p.font.color.rgb = CLR_TEXT_SEC; p.space_before = Pt(4)

    # Arrow 2
    ar = slide7.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(6.35), dy7 + Inches(0.6), Inches(0.45), Inches(0.3))
    ar.fill.solid(); ar.fill.fore_color.rgb = CLR_PRIMARY; ar.line.fill.background()

    # Step 3: Voxel Accumulator
    add_card(slide7, Inches(6.9), dy7, dw7, dh7, CLR_SHADOW_BOX, CLR_CARD_BORDER)
    tb = slide7.shapes.add_textbox(Inches(6.95), dy7 + Inches(0.1), dw7 - Inches(0.1), dh7 - Inches(0.2))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "3. Voxel Accumulator"; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = CLR_ACCENT_BLUE
    p = tf.add_paragraph(); p.text = "• 3D occupancy grid\n• Gaussian weight fusion\n• Uncertainty tracking"; p.font.size = Pt(10); p.font.color.rgb = CLR_TEXT_SEC; p.space_before = Pt(4)

    # Arrow 3
    ar = slide7.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(9.25), dy7 + Inches(0.6), Inches(0.45), Inches(0.3))
    ar.fill.solid(); ar.fill.fore_color.rgb = CLR_PRIMARY; ar.line.fill.background()

    # Step 4: Next-Best-View Planner
    add_card(slide7, Inches(9.8), dy7, dw7, dh7, CLR_SHADOW_BOX, CLR_CARD_BORDER)
    tb = slide7.shapes.add_textbox(Inches(9.85), dy7 + Inches(0.1), dw7 - Inches(0.1), dh7 - Inches(0.2))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "4. Active NBV Planner"; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = CLR_ACCENT_AMBER
    p = tf.add_paragraph(); p.text = "• Submodular gain\n• Maximize coverage\n• Minimize shadow voids"; p.font.size = Pt(10); p.font.color.rgb = CLR_TEXT_SEC; p.space_before = Pt(4)

    # Write-up: 3 Small Cards (Bottom Half)
    # Card 1: Physical Acoustic Ray Slicing
    add_card(slide7, start_left, b_top, card_w3, b_h)
    tb_b1 = slide7.shapes.add_textbox(start_left + Inches(0.15), b_top + Inches(0.15), card_w3 - Inches(0.3), b_h - Inches(0.3))
    tf_b1 = tb_b1.text_frame; tf_b1.word_wrap = True
    p = tf_b1.paragraphs[0]; p.text = "Physical Wave Simulation"; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = CLR_PRIMARY
    p = tf_b1.add_paragraph(); p.text = "\n• Why not pre-rendered images?\nReal ultrasound exhibits physical wave behaviors that change dynamically with probe tilt and pressure."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_SEC
    p = tf_b1.add_paragraph(); p.text = "\n• GPU PyTorch Layer:\nComputes depth-dependent acoustic attenuation, impedance boundary reflections, and Rayleigh speckle noise in parallel."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_SEC

    # Card 2: Confidence-Weighted Voxel Grid
    add_card(slide7, start_left + card_w3 + col_gap, b_top, card_w3, b_h)
    tb_b2 = slide7.shapes.add_textbox(start_left + card_w3 + col_gap + Inches(0.15), b_top + Inches(0.15), card_w3 - Inches(0.3), b_h - Inches(0.3))
    tf_b2 = tb_b2.text_frame; tf_b2.word_wrap = True
    p = tf_b2.paragraphs[0]; p.text = "Voxel Confidence Fusion"; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = CLR_ACCENT_BLUE
    p = tf_b2.add_paragraph(); p.text = "\n• Probabilistic Fusion:\nFuses successive 2D sweeps into a continuous 3D volumetric tensor using Gaussian spatial weighting."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_SEC
    p = tf_b2.add_paragraph(); p.text = "\n• Uncertainty Quantification:\nAssigns lower confidence to deeper rays and acoustic shadow regions behind bone or dense tissue."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_SEC

    # Card 3: Active Next-Best-View (NBV)
    add_card(slide7, start_left + (card_w3 + col_gap)*2, b_top, card_w3, b_h)
    tb_b3 = slide7.shapes.add_textbox(start_left + (card_w3 + col_gap)*2 + Inches(0.15), b_top + Inches(0.15), card_w3 - Inches(0.3), b_h - Inches(0.3))
    tf_b3 = tb_b3.text_frame; tf_b3.word_wrap = True
    p = tf_b3.paragraphs[0]; p.text = "Active Next-Best-View"; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = CLR_ACCENT_AMBER
    p = tf_b3.add_paragraph(); p.text = "\n• Autonomous Path Planning:\nRather than blind raster scanning, the robot dynamically evaluates candidate 6-DoF probe poses."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_SEC
    p = tf_b3.add_paragraph(); p.text = "\n• Information-Theoretic Gain:\nSelects the next trajectory that maximizes volumetric entropy reduction while minimizing robot travel time."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_SEC

    slide7.notes_slide.notes_text_frame.text = (
        "Slide 7 Speaker Notes:\n"
        "Here is our second major accomplished pillar: The 3D Ultrasound Reconstruction Pipeline.\n\n"
        "Let's trace the flow diagram across the top:\n"
        "We start with a patient CT volume in Hounsfield units. As the Kinova arm positions the probe, PyTorch's `grid_sample` operator extracts an arbitrary 2D planar slice in GPU memory without CPU round trips.\n\n"
        "Next, our acoustic shader applies three core physical wave phenomena: First, exponential depth attenuation—ultrasound loses energy as it travels deeper. Second, acoustic impedance reflections at tissue boundaries. Third, Rayleigh scattering noise to reproduce realistic speckle patterns.\n\n"
        "These 2D B-scans then feed into our 3D Voxel Accumulator, which updates occupancy and uncertainty grids.\n\n"
        "Finally, the Next-Best-View planner evaluates which unobserved or high-uncertainty regions need to be scanned next, commanding the Kinova arm to adjust its tilt and position to fill in acoustic shadows. This provides a completely autonomous scanning loop."
    )

    # =========================================================================
    # SLIDE 8: TECHNICAL UNDERSTANDING: KEY ARCHITECTURAL DECISIONS
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide8)
    add_header(slide8, "4. Technical Understanding", "Architectural Decisions & Engineering Trade-Offs",
               "Evaluating key design choices, rationale, and answers to fundamental technical questions")

    # 3 Large Comparative Cards
    dec_w = Inches(3.64)
    dec_h = Inches(4.8)
    dec_top = Inches(1.8)

    # Decision 1: OpenIGTLink vs ROS2
    add_card(slide8, start_left, dec_top, dec_w, dec_h)
    tb_d1 = slide8.shapes.add_textbox(start_left + Inches(0.2), dec_top + Inches(0.2), dec_w - Inches(0.4), dec_h - Inches(0.4))
    tf_d1 = tb_d1.text_frame; tf_d1.word_wrap = True
    p = tf_d1.paragraphs[0]; p.text = "DECISION 1"
    p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = CLR_PRIMARY
    p = tf_d1.add_paragraph(); p.text = "OpenIGTLink vs. ROS2"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = CLR_DARK; p.space_before = Pt(4)

    p = tf_d1.add_paragraph(); p.text = "\n• The Trade-Off:\nROS2 provides great tooling for general robotics, but its DDS middleware introduces significant latency, complex cross-compilation, and OS dependency locks."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_SEC

    p = tf_d1.add_paragraph(); p.text = "\n• Why OpenIGTLink?\n1. Native Clinical Standard: 3D Slicer natively parses OpenIGTLink binary messages without intermediate bridges."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_PRI

    p = tf_d1.add_paragraph(); p.text = "\n2. Sub-Millisecond Jitter: Direct TCP binary sockets eliminate serialization bottlenecks during 1000 Hz physics loops."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_PRI

    # Decision 2: Model-Based GPU Acoustics vs NeRF/Diffusion
    add_card(slide8, start_left + dec_w + col_gap, dec_top, dec_w, dec_h)
    tb_d2 = slide8.shapes.add_textbox(start_left + dec_w + col_gap + Inches(0.2), dec_top + Inches(0.2), dec_w - Inches(0.4), dec_h - Inches(0.4))
    tf_d2 = tb_d2.text_frame; tf_d2.word_wrap = True
    p = tf_d2.paragraphs[0]; p.text = "DECISION 2"
    p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = CLR_ACCENT_BLUE
    p = tf_d2.add_paragraph(); p.text = "GPU Acoustics vs. Neural Rendering"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = CLR_DARK; p.space_before = Pt(4)

    p = tf_d2.add_paragraph(); p.text = "\n• The Trade-Off:\nNeural Radiance Fields (NeRFs) and diffusion generators produce photorealistic textures but suffer from hallucination and require hours of pre-training per patient scan."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_SEC

    p = tf_d2.add_paragraph(); p.text = "\n• Why Model-Based GPU Physics?\n1. Instant Generalization: Runs zero-shot on ANY new patient CT scan in milliseconds via PyTorch tensor slicing."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_PRI

    p = tf_d2.add_paragraph(); p.text = "\n2. Strict Physical Validity: Acoustic wave attenuation and acoustic shadows are mathematically exact rather than hallucinated."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_PRI

    # Decision 3: Submodular NBV vs Heuristic Raster Scanning
    add_card(slide8, start_left + (dec_w + col_gap)*2, dec_top, dec_w, dec_h)
    tb_d3 = slide8.shapes.add_textbox(start_left + (dec_w + col_gap)*2 + Inches(0.2), dec_top + Inches(0.2), dec_w - Inches(0.4), dec_h - Inches(0.4))
    tf_d3 = tb_d3.text_frame; tf_d3.word_wrap = True
    p = tf_d3.paragraphs[0]; p.text = "DECISION 3"
    p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = CLR_ACCENT_AMBER
    p = tf_d3.add_paragraph(); p.text = "Submodular NBV vs. Raster Sweeps"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = CLR_DARK; p.space_before = Pt(4)

    p = tf_d3.add_paragraph(); p.text = "\n• The Trade-Off:\nFixed raster/zigzag sweeping is simple to script but wastes massive time scanning already-observed tissue while missing occluded anatomy."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_SEC

    p = tf_d3.add_paragraph(); p.text = "\n• Why Submodular Information Gain?\n1. Minimal Scan Duration: Prioritizes views that maximize volumetric entropy reduction."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_PRI

    p = tf_d3.add_paragraph(); p.text = "\n2. Shadow Disambiguation: Intelligently tilts the probe around acoustic obstacles (like ribs or bone) to illuminate hidden tissue."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_PRI

    slide8.notes_slide.notes_text_frame.text = (
        "Slide 8 Speaker Notes:\n"
        "This slide directly demonstrates Technical Understanding—the ability to defend our architectural choices under committee scrutiny.\n\n"
        "Decision 1: Why OpenIGTLink over ROS2? A common question is 'Why didn't you just use ROS2 topics?'. The answer is two-fold: First, clinical reality. Surgical operating rooms don't run ROS; they use OpenIGTLink-enabled devices. Second, real-time performance. Serializing large arrays into ROS2 DDS introduces noticeable jitter, whereas our direct TCP binary socket transmits state updates in under 0.5 milliseconds.\n\n"
        "Decision 2: Why Model-Based Acoustic Physics instead of a Generative AI / NeRF renderer? NeRFs and diffusion models are prone to hallucinating non-existent tissue features—a fatal flaw in healthcare. Moreover, they require retraining for every new patient anatomy. Our GPU PyTorch shader runs instantly on any patient CT scan and strictly obeys acoustic physics equations.\n\n"
        "Decision 3: Why Submodular Next-Best-View planning over conventional raster scanning? Raster scanning scans empty space and repeatedly sweeps flat areas while leaving acoustic shadows unaddressed. Our submodular entropy planner specifically tilts the probe to look around acoustic obstacles."
    )

    # =========================================================================
    # SLIDE 9: FUTURE DIRECTION: LEARNING & BENCHMARKING
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide9)
    add_header(slide9, "5. Future Direction", "Policy Learning & Sim-to-Real Roadmap",
               "Next steps for the second half of the project: VLA integration, policy benchmarking, and phantom transfer")

    # 3 Milestone Cards (Timeline Layout)
    m_w = Inches(3.64)
    m_h = Inches(4.8)
    m_top = Inches(1.8)

    # Phase 1: VLA Integration
    add_card(slide9, start_left, m_top, m_w, m_h)
    tb_m1 = slide9.shapes.add_textbox(start_left + Inches(0.2), m_top + Inches(0.2), m_w - Inches(0.4), m_h - Inches(0.4))
    tf_m1 = tb_m1.text_frame; tf_m1.word_wrap = True
    p = tf_m1.paragraphs[0]; p.text = "PHASE 1 (WEEKS 9-11)"
    p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = CLR_PRIMARY
    p = tf_m1.add_paragraph(); p.text = "Multimodal VLA & Policy Learning"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = CLR_DARK; p.space_before = Pt(4)

    p = tf_m1.add_paragraph(); p.text = "\n• Action-Chunking Diffusion Policies:\nDeploy Continuous Flow Matching (CFM) and diffusion action policies for smooth 6-DoF probe velocity generation."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_SEC

    p = tf_m1.add_paragraph(); p.text = "\n• Multimodal Conditioning:\nCondition the policy on synthesized B-mode ultrasound, contact force feedback, and natural language clinical goals (e.g., 'Reconstruct hepatic vein')."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_SEC

    # Phase 2: Benchmarking Suite
    add_card(slide9, start_left + m_w + col_gap, m_top, m_w, m_h)
    tb_m2 = slide9.shapes.add_textbox(start_left + m_w + col_gap + Inches(0.2), m_top + Inches(0.2), m_w - Inches(0.4), m_h - Inches(0.4))
    tf_m2 = tb_m2.text_frame; tf_m2.word_wrap = True
    p = tf_m2.paragraphs[0]; p.text = "PHASE 2 (WEEKS 12-14)"
    p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = CLR_ACCENT_BLUE
    p = tf_m2.add_paragraph(); p.text = "Benchmarking & Evaluation Suite"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = CLR_DARK; p.space_before = Pt(4)

    p = tf_m2.add_paragraph(); p.text = "\n• Standardized Benchmark Tasks:\nEvaluate policy performance across a dataset of 20+ diverse patient abdominal and cardiac CT scans."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_SEC

    p = tf_m2.add_paragraph(); p.text = "\n• Evaluation Metrics:\nMeasure volumetric coverage percentage, acoustic shadow volume, total scan trajectory length, and contact force boundary adherence."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_SEC

    # Phase 3: Hardware Sim-to-Real Transfer
    add_card(slide9, start_left + (m_w + col_gap)*2, m_top, m_w, m_h)
    tb_m3 = slide9.shapes.add_textbox(start_left + (m_w + col_gap)*2 + Inches(0.2), m_top + Inches(0.2), m_w - Inches(0.4), m_h - Inches(0.4))
    tf_m3 = tb_m3.text_frame; tf_m3.word_wrap = True
    p = tf_m3.paragraphs[0]; p.text = "PHASE 3 (WEEKS 15-16)"
    p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = CLR_ACCENT_AMBER
    p = tf_m3.add_paragraph(); p.text = "Sim-to-Real Hardware Transfer"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = CLR_DARK; p.space_before = Pt(4)

    p = tf_m3.add_paragraph(); p.text = "\n• Phantom Validation:\nDeploy the trained autonomous scanning policy onto a physical 7-DoF robotic manipulator scanning an ultrasound phantom."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_SEC

    p = tf_m3.add_paragraph(); p.text = "\n• Domain Randomization:\nApply acoustic impedance and friction randomization during simulation training to achieve zero-shot transfer."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_SEC

    slide9.notes_slide.notes_text_frame.text = (
        "Slide 9 Speaker Notes:\n"
        "Turning to Future Direction: What are the concrete, feasible next steps for the remaining half of the project?\n\n"
        "Phase 1 (Weeks 9 to 11): We will train imitation learning and diffusion-based action-chunking policies (such as Continuous Flow Matching). The policy will take in multimodal tokens: the synthesized ultrasound frame, current contact forces, and high-level language prompts specifying target anatomical regions.\n\n"
        "Phase 2 (Weeks 12 to 14): We will run systematic benchmarking across diverse patient CT scans, measuring coverage completeness, scan time, and force safety margins.\n\n"
        "Phase 3 (Weeks 15 to 16): Hardware Sim-to-Real transfer. Because our simulation models physical contact dynamics and ray-traced acoustics, we will validate the policy on a physical robot scanning a silicone phantom to demonstrate real-world applicability."
    )

    # =========================================================================
    # SLIDE 10: CONCLUSION & SUMMARY
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide10)
    add_header(slide10, "Conclusion", "Project Summary & Key Takeaways",
               "Recap of technical accomplishments, project impact, and readiness for AI benchmarking")

    # 3 Summary Cards
    sum_w = Inches(3.64)
    sum_h = Inches(3.4)
    sum_top = Inches(1.8)

    # Card 1: Core Platform Built
    add_card(slide10, start_left, sum_top, sum_w, sum_h)
    tb_s1 = slide10.shapes.add_textbox(start_left + Inches(0.2), sum_top + Inches(0.2), sum_w - Inches(0.4), sum_h - Inches(0.4))
    tf_s1 = tb_s1.text_frame; tf_s1.word_wrap = True
    p = tf_s1.paragraphs[0]; p.text = "CORE PLATFORM BUILT"
    p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = CLR_PRIMARY
    p = tf_s1.add_paragraph(); p.text = "Slicer Surgical Bridge"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = CLR_DARK; p.space_before = Pt(4)
    p = tf_s1.add_paragraph(); p.text = "\n• Functional OpenIGTLink binary link connecting 1000 Hz PyBullet physics to 3D Slicer at 60 FPS.\n• Real-time virtual keep-out safety fixtures preventing patient harm."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_SEC

    # Card 2: Physical Acoustic Pipeline
    add_card(slide10, start_left + sum_w + col_gap, sum_top, sum_w, sum_h)
    tb_s2 = slide10.shapes.add_textbox(start_left + sum_w + col_gap + Inches(0.2), sum_top + Inches(0.2), sum_w - Inches(0.4), sum_h - Inches(0.4))
    tf_s2 = tb_s2.text_frame; tf_s2.word_wrap = True
    p = tf_s2.paragraphs[0]; p.text = "PHYSICAL WAVE SYNTHESIS"
    p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = CLR_ACCENT_BLUE
    p = tf_s2.add_paragraph(); p.text = "3D Reconstruction Pipeline"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = CLR_DARK; p.space_before = Pt(4)
    p = tf_s2.add_paragraph(); p.text = "\n• GPU PyTorch reslicing with physical acoustic attenuation, impedance reflections, and speckle.\n• Active submodular Next-Best-View planning to eliminate acoustic shadows."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_SEC

    # Card 3: Ready for AI Benchmarking
    add_card(slide10, start_left + (sum_w + col_gap)*2, sum_top, sum_w, sum_h)
    tb_s3 = slide10.shapes.add_textbox(start_left + (sum_w + col_gap)*2 + Inches(0.2), sum_top + Inches(0.2), sum_w - Inches(0.4), sum_h - Inches(0.4))
    tf_s3 = tb_s3.text_frame; tf_s3.word_wrap = True
    p = tf_s3.paragraphs[0]; p.text = "AI BENCHMARK READY"
    p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = CLR_ACCENT_AMBER
    p = tf_s3.add_paragraph(); p.text = "Sim-to-Real Foundation"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = CLR_DARK; p.space_before = Pt(4)
    p = tf_s3.add_paragraph(); p.text = "\n• Clean APIs ready for training Vision-Language-Action policies and diffusion controllers.\n• A solid, reproducible foundation for final evaluation and sim-to-real transfer."
    p.font.size = Pt(11); p.font.color.rgb = CLR_TEXT_SEC

    # Bottom Discussion / Acknowledgement Card
    disc_card = add_card(slide10, Inches(0.8), Inches(5.4), Inches(11.733), Inches(1.5), CLR_SHADOW_BOX, CLR_CARD_BORDER)
    tb_disc = slide10.shapes.add_textbox(Inches(1.1), Inches(5.5), Inches(11.1), Inches(1.3))
    tf_disc = tb_disc.text_frame; tf_disc.word_wrap = True
    p = tf_disc.paragraphs[0]; p.text = "THANK YOU  |  QUESTIONS & DISCUSSION"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = CLR_PRIMARY; p.alignment = PP_ALIGN.CENTER
    p = tf_disc.add_paragraph()
    p.text = "Candidate: Rishang Yadav (B23173)   |   Supervisor: Dr. Deepak Raina\nSchool of Computing and Electrical Engineering (SCEE), IIT Mandi"
    p.font.size = Pt(12); p.font.color.rgb = CLR_DARK; p.alignment = PP_ALIGN.CENTER; p.space_before = Pt(6)

    slide10.notes_slide.notes_text_frame.text = (
        "Slide 10 Speaker Notes:\n"
        "To conclude our mid-term evaluation:\n"
        "Over the past two months, we have built the core foundational infrastructure for contact-rich medical robotics simulation.\n"
        "We have proven that OpenIGTLink provides an ultra-low latency, hospital-standard link directly into 3D Slicer, completely superseding heavy ROS middleware. We have developed a GPU-accelerated acoustic physics shader that replicates real ultrasound wave mechanics without black-box neural hallucinations. And we have established an active Next-Best-View planning loop that automates volumetric scanning.\n\n"
        "With this foundation fully operational, we are on track to train multimodal VLA policies and benchmark sim-to-real transfer over the coming weeks.\n\n"
        "Thank you for your time, and I am now ready to take any questions and discuss your feedback."
    )

    # Save presentation
    output_pptx = "/home/rishang/Desktop/Insti_Stuff/MTP/mtp_mid_evaluation.pptx"
    prs.save(output_pptx)
    print(f"Presentation successfully saved to: {output_pptx}")

if __name__ == '__main__':
    create_deck()
