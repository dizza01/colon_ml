# 🔍 Explainability Metrics for Medical AI

This document explains the key explainability metrics used in our polyp detection system, particularly focusing on Quantus metrics for evaluating explanation quality.

## 📊 Current Implementation

Our system currently uses **Sparseness** from the Quantus library to evaluate explanation quality:

```python
from quantus import Sparseness

sparseness_metric = Sparseness()
score = sparseness_metric(model=model, x_batch=input_np, y_batch=y_target, a_batch=attr)
```

## 🎯 Core Explainability Metrics

### **Sparseness**

**What it measures**: How "focused" or "scattered" the explanation is

**Simple analogy**: Like a flashlight beam
- **High sparseness (0.9+)**: Narrow, focused beam highlighting just the polyp
- **Low sparseness (0.3-)**: Wide, scattered beam highlighting the entire image

**For polyp detection**:
```python
# Good explanation (high sparseness):
# 🔴🔴⚫⚫⚫  (highlights only polyp pixels)
# ⚫⚫⚫⚫⚫
# ⚫⚫⚫⚫⚫

# Poor explanation (low sparseness):
# 🟡🟡🟡🟡🟡  (highlights everything)
# 🟡🟡🟡🟡🟡
# 🟡🟡🟡🟡🟡
```

**Clinical benefit**: Helps doctors focus on the exact polyp location, not the entire image

---

### **Max Sensitivity**

**What it measures**: How much the explanation changes when you make tiny changes to the input image

**Simple analogy**: Like a nervous person vs. a calm person
- **High sensitivity**: Explanation changes dramatically with small input noise (unreliable)
- **Low sensitivity**: Explanation stays consistent despite small changes (reliable)

**For polyp detection**:
```python
# Original image: Polyp at position (50, 60)
# Add tiny noise (1 pixel brighter)

# Good explanation (low sensitivity):
# Still highlights polyp at (50, 60) ✅

# Poor explanation (high sensitivity):  
# Now highlights completely different area! ❌
```

**Clinical benefit**: Ensures explanations are trustworthy - won't change based on image quality variations

---

### **Faithfulness**

**What it measures**: Whether the "important" pixels actually matter for the model's decision

**Simple analogy**: Like testing if someone really listens to your advice
- **High faithfulness**: When you remove "important" pixels, prediction changes a lot
- **Low faithfulness**: Removing "important" pixels doesn't affect prediction much

**For polyp detection**:
```python
# Test process:
# 1. Model says: "This pixel is crucial for detecting polyp"
# 2. Remove that pixel and re-run model
# 3. Check: Did prediction change significantly?

# High faithfulness:
# Remove highlighted pixel → Prediction drops from 0.9 to 0.3 ✅
# (The explanation was truthful)

# Low faithfulness:
# Remove highlighted pixel → Prediction stays at 0.9 ❌
# (The explanation was misleading)
```

**Clinical benefit**: Ensures the model actually uses the features it claims to use

## 🏥 Clinical Summary

| Metric | Clinical Question | Good Score Means | Implementation Status |
|--------|------------------|------------------|---------------------|
| **Sparseness** | "Is the explanation focused?" | Doctor sees exact polyp location | ✅ **Implemented** |
| **Max Sensitivity** | "Is the explanation reliable?" | Same explanation across image variations | 🔄 **Recommended** |
| **Faithfulness** | "Is the explanation truthful?" | Model actually uses highlighted features | 🔄 **Recommended** |

## 🔬 Additional Quantus Metrics Available

### **Localization Quality**
```python
from quantus import PointingGame, TopKIntersection, RelevanceRankAccuracy
```

### **Robustness & Reliability**  
```python
from quantus import MaxSensitivity, AvgSensitivity, LocalLipschitzEstimate
```

### **Faithfulness**
```python
from quantus import Faithfulness, PixelFlipping, RegionPerturbation
```

### **Complexity**
```python
from quantus import Complexity, EffectiveComplexity
```

## 💡 Recommendations for Medical Imaging

For polyp detection, these metrics would be particularly valuable:

1. **PointingGame**: Does the explanation point to actual polyp regions?
2. **Faithfulness**: Do important pixels actually affect the prediction?
3. **MaxSensitivity**: How stable are explanations to small input changes?

## 🎯 Key Benefits for Clinical Practice

These metrics help ensure your AI explanations are:

- **🎯 Focused**: Precise highlighting of relevant anatomical features
- **🔒 Stable**: Consistent explanations across imaging conditions
- **✅ Honest**: Model actually uses the highlighted features for decisions

This comprehensive evaluation framework is critical for building trust in medical AI systems and ensuring safe clinical deployment.


