# OpenDrive-XAI Implementation Roadmap

> **⚠️ Python 3.12 Required:**
> This project currently requires **Python 3.12** due to dependency compatibility (notably pydantic and CARLA). Python 3.13+ is not yet supported. Please ensure you are using Python 3.12 when setting up your environment.

## **Current Status Analysis**

### ✅ **Completed Infrastructure**
- [x] Basic Python package structure with CI/CD
- [x] Configuration management (Pydantic-based)
- [x] Logging utilities
- [x] Simple BEV encoder (ResNet18 placeholder)
- [x] CARLA sample data pipeline
- [x] CARLA simulation environment interface
- [x] Vision Transformer architecture (advanced)
- [x] Multi-camera BEV projection system
- [x] Training pipeline framework

### 🚧 **Current Phase: 0.1 - Lane-keeping MVP**
**Target: August 2025** | **Current Progress: ~40%**

---

## **IMMEDIATE NEXT STEPS (Next 4-8 weeks)**

### **Phase 1A: Complete Core Infrastructure** ⭐ **HIGH PRIORITY**

#### 1. **Fix Vision Transformer Integration** (Week 1)
```bash
# Action Items:
- Fix linter errors in vision_transformer.py
- Create comprehensive unit tests for MultiCameraBEVTransformer
- Implement proper attention weight extraction for XAI
- Add model size optimization for CPU inference
```

**Implementation:**
```python
# Create test file
OpenDrive-XAI/tests/test_vision_transformer.py

# Create model checkpoint utilities
OpenDrive-XAI/src/opendrive_xai/models/checkpoint_utils.py

# Create XAI visualization tools
OpenDrive-XAI/src/opendrive_xai/explainability/attention_maps.py
```

#### 2. **Create Training Data Pipeline** (Week 1-2)
```bash
# Action Items:
- Implement real CARLA data collection script
- Create data preprocessing pipeline (image normalization, augmentation)
- Build dataset class for multi-camera temporal sequences
- Add data validation and quality checks
```

**Key Files to Create:**
- `scripts/collect_carla_data.py` - Automated data collection
- `src/opendrive_xai/data/preprocessing.py` - Data preprocessing
- `src/opendrive_xai/data/validation.py` - Data quality checks

#### 3. **Setup CARLA Integration** (Week 2)
```bash
# Action Items:
- Create CARLA Docker setup for consistent environment
- Implement lane-keeping scenario in CARLA
- Add expert trajectory collection pipeline
- Create evaluation metrics for lane-keeping
```

### **Phase 1B: MVP Implementation** ⭐ **CRITICAL PATH**

#### 4. **Train Initial Lane-Keeping Model** (Week 3-4)
```bash
# Action Items:
- Collect 50+ hours of CARLA lane-keeping data
- Train simplified 2-camera ViT model (front + rear)
- Implement basic trajectory prediction loss
- Add real-time inference pipeline
```

**Expected Deliverables:**
- Trained model achieving <2m trajectory error in simulation
- Real-time inference pipeline (>10 FPS)
- Basic visualization dashboard

#### 5. **Create Evaluation Framework** (Week 4)
```bash
# Action Items:
- Implement closed-loop CARLA evaluation
- Add safety metrics (collision rate, route completion)
- Create comparison with rule-based baseline
- Add performance benchmarking tools
```

---

## **MEDIUM-TERM MILESTONES (Next 2-6 months)**

### **Phase 0.2: Multi-sensor Perception** (Target: Oct 2025)

#### **Multi-Modal Sensor Fusion**
- Add LiDAR sensor support to CARLA environment
- Implement point cloud processing pipeline
- Create camera-LiDAR fusion architecture
- Add object detection capabilities

#### **Advanced Perception**
- Integrate radar sensor simulation
- Implement multi-object tracking (UKF/Particle Filters)
- Add depth estimation from stereo cameras
- Create semantic segmentation pipeline

### **Phase 0.3: Robust Planning** (Target: Jan 2026)

#### **Advanced Planning**
- Implement MPC + Diffusion planner (MAD)
- Add behavioral prediction for other agents
- Create adversarial scenario test suite
- Integrate causal inference modules

#### **Robustness Features**
- Add adversarial training pipeline
- Implement domain randomization
- Create out-of-distribution detection
- Add safety constraint integration

---

## **TECHNICAL IMPLEMENTATION DETAILS**

### **1. Immediate Development Environment Setup**

```