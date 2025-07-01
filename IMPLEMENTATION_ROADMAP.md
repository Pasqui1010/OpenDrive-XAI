# OpenDrive-XAI Implementation Roadmap

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

```bash
# Install CARLA (required for data collection)
cd OpenDrive-XAI
./scripts/setup_carla.sh

# Setup development environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# Download extended training data
./scripts/download_training_data.sh
```

### **2. Critical Missing Components**

#### **A. Data Collection Pipeline**
```python
# scripts/collect_training_data.py
"""
Automated CARLA data collection with:
- Multi-camera setup (6 cameras)
- Expert driver demonstrations
- Diverse weather/lighting conditions
- Edge case scenario generation
"""
```

#### **B. Model Training Script**
```python
# scripts/train_lane_keeping.py
"""
End-to-end training script with:
- Multi-GPU support
- Checkpoint management
- Wandb integration
- Real-time validation
"""
```

#### **C. Real-time Inference**
```python
# src/opendrive_xai/inference/realtime_pipeline.py
"""
Real-time inference pipeline with:
- <100ms latency requirement
- Multi-camera synchronization
- Safety fallback mechanisms
- Performance monitoring
"""
```

### **3. Testing Strategy**

#### **Unit Tests (Immediate)**
- Vision Transformer forward/backward pass
- CARLA environment connection/control
- Data pipeline integrity
- Inference latency benchmarks

#### **Integration Tests (Week 2-3)**
- End-to-end training pipeline
- CARLA closed-loop evaluation
- Multi-camera data synchronization
- Model checkpoint loading/saving

#### **System Tests (Week 4)**
- Lane-keeping performance in CARLA
- Robustness to weather variations
- Comparison with expert demonstrations
- Real-time performance validation

---

## **RESOURCE REQUIREMENTS**

### **Computational Resources**
- **Training**: NVIDIA RTX 4080/A100 (cloud recommended)
- **Development**: RTX 3080+ or cloud instances
- **CARLA Simulation**: 16GB+ RAM, modern CPU

### **Data Storage**
- **Phase 0.1**: ~500GB (50 hours × 6 cameras × 20 FPS)
- **Phase 0.2**: ~2TB (multi-sensor fusion data)
- **Cloud Storage**: AWS S3 or Google Cloud Storage

### **Team Assignments**
Based on your comprehensive project plan:
- **ML Architect**: Vision Transformer optimization & XAI
- **Simulation Engineer**: CARLA environment & scenarios
- **Data Pipeline Specialist**: Data collection & preprocessing
- **Safety Specialist**: Evaluation metrics & robustness

---

## **SUCCESS CRITERIA FOR PHASE 0.1**

### **Functional Requirements**
- [x] ✅ Multi-camera data processing
- [ ] 🔄 Lane-keeping in CARLA simulation
- [ ] 🔄 <2m trajectory error on test routes
- [ ] 🔄 Real-time inference (>10 FPS)
- [ ] 🔄 Basic explainability (attention maps)

### **Technical Requirements**
- [ ] 🔄 Docker-based development environment
- [ ] 🔄 Comprehensive unit test coverage (>80%)
- [ ] 🔄 Automated CI/CD pipeline
- [ ] 🔄 Documentation & API reference

### **Performance Benchmarks**
- **Route Completion Rate**: >95% in clear weather
- **Collision Rate**: <1% in normal traffic
- **Inference Latency**: <100ms end-to-end
- **Model Size**: <500MB for deployment

---

## **RISK MITIGATION**

### **High-Risk Items**
1. **Sim-to-Real Gap**: Mitigate with domain randomization
2. **Training Data Quality**: Implement automated validation
3. **Real-time Performance**: Optimize model architecture early
4. **CARLA Stability**: Use containerized environments

### **Fallback Plans**
- **Model Complexity**: Start with simplified 2-camera setup
- **Training Data**: Use synthetic augmentation if collection fails
- **Performance**: Implement model distillation for edge deployment

---

## **IMMEDIATE ACTION ITEMS (This Week)**

### **Day 1-2: Environment Setup**
```bash
# 1. Setup CARLA development environment
git clone https://github.com/carla-simulator/carla.git
cd carla && ./setup.sh

# 2. Create data collection script
touch scripts/collect_carla_training_data.py

# 3. Fix vision transformer implementation
# - Address linter errors
# - Add comprehensive tests
# - Validate model architecture
```

### **Day 3-5: Initial Data Collection**
```bash
# 1. Collect initial dataset (10+ hours)
python scripts/collect_carla_training_data.py --hours 10 --scenarios lane_keeping

# 2. Validate data quality
python scripts/validate_training_data.py --data_dir data/training

# 3. Train initial model
python scripts/train_lane_keeping.py --config configs/mvp.yaml
```

### **Week 2: Model Integration & Testing**
- Integrate trained model with CARLA evaluation
- Create real-time inference pipeline
- Add basic visualization dashboard
- Conduct preliminary performance evaluation

---

**Next Review**: End of Week 2 - Assess MVP progress and adjust timeline based on initial results. 