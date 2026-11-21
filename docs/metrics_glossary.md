# Evaluation Metrics Glossary & Batch Inference Pipeline

## 1. Segmentation Evaluation Metrics Glossary

### A. Dice Similarity Coefficient (DSC / F1-Score)
- **Mathematical Formula:**
  $$\text{Dice}(A, B) = \frac{2 |A \cap B|}{|A| + |B|} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$
- **Range:** $[0.0, 1.0]$ ($0\%$ to $100\%$). Higher is better ($1.0$ is perfect overlap).
- **Physical Meaning:** Measures region-level overlap between predicted lesion mask $A$ and ground-truth expert mask $B$. Gives double weight to True Positives (correctly segmented lesion area).
- **Clinical Nuance:** Great for volumetric accuracy, but can mask boundary contour errors on large lesions (e.g. jagged edges on a large tumor might still yield a $>90\%$ Dice score).

---

### B. Jaccard Index (Intersection over Union / IoU)
- **Mathematical Formula:**
  $$\text{Jaccard}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$$
- **Relationship to Dice:**
  $$\text{Jaccard} = \frac{\text{Dice}}{2 - \text{Dice}}$$
- **Range:** $[0.0, 1.0]$. Higher is better.
- **Physical Meaning:** Stricter penalization of False Positives (over-segmentation) and False Negatives (under-segmentation) than Dice.

---

### C. 95% Hausdorff Distance (95HD / HD95)
- **Mathematical Formula:**
  $$95\text{HD}(A, B) = \text{Percentile}_{95\%} \left( \min_{y \in \partial B} \|x - y\|, \, \min_{x \in \partial A} \|x - y\| \right)$$
- **Unit:** Distance in pixels (or millimeters). Lower is better ($0.0$ indicates perfect boundary alignment).
- **Why 95% instead of 100% (Maximum Hausdorff Distance)?**
  - Standard $100\%$ Hausdorff distance takes the single maximum distance point. A single spurious false-positive pixel 50 pixels away from the tumor will blow up $100\text{HD}$ to $50.0$.
  - Taking the **95th percentile** eliminates boundary outliers while measuring the realistic worst-case margin error along the lesion perimeter.

---

### D. Average Surface Distance (ASD)
- **Mathematical Formula:**
  $$\text{ASD}(A, B) = \frac{1}{|\partial A| + |\partial B|} \left( \sum_{x \in \partial A} \min_{y \in \partial B} \|x - y\| + \sum_{y \in \partial B} \min_{x \in \partial A} \|x - y\| \right)$$
- **Unit:** Distance in pixels (or millimeters). Lower is better.
- **Physical Meaning:** The average distance/thickness error from every boundary pixel of the prediction to the nearest ground truth boundary pixel.

---

## 2. Clinical & Ultrasound Terminology

| Term | Category | Description |
| :--- | :--- | :--- |
| **SRAD** | Preprocessing | *Speckle Reducing Anisotropic Diffusion*. Solves a partial differential equation (PDE) to smooth homogeneous multiplicative speckle noise while preserving sharp lesion boundaries. |
| **CLAHE** | Preprocessing | *Contrast Limited Adaptive Histogram Equalization*. Enhances local tissue contrast in small contextual tiles with a clip limit to prevent over-amplifying background noise. |
| **DAG** | Architecture | *Dual Attention Gate*. Applied to decoder skip connections. Combines Global Average Pooling and Global Max Pooling to filter non-informative background features. |
| **MFEF** | Architecture | *Multi-Scale Feature Extraction and Fusion*. Bottleneck module combining $1\times1$, $3\times3$, $5\times5$, global context, and dilated convs ($d=2$) with Softmax channel attention. |
| **BI-RADS** | Clinical | *Breast Imaging-Reporting and Data System*. Standardized radiological scoring system for lesion risk assessment. |

---

## 3. Progress Update: Streamlit Batch Demo (`G17-Model`)

### Key Implemented Features:
1. **Multi-Image Batch Upload:** Enabled `accept_multiple_files=True` allowing examiners to upload multiple ultrasound scans at once.
2. **Concurrent Preprocessing:** Dispatched SRAD and CLAHE across worker threads using `ThreadPoolExecutor` to prevent blocking the UI.
3. **Batched PyTorch Forward Pass:** Stacked $N$ preprocessed tensors into a single `(B, 1, 256, 256)` tensor for 1 forward pass on CPU.
4. **Instant Threshold Recalculation:** Cached raw probability maps in `st.session_state`. Adjusting the probability threshold slider re-slices binary masks and overlays in $<1\text{ ms}$ without re-running the model or SRAD.
5. **Batch Summary Metrics:** Added a dataframe table displaying scan dimensions, lesion detection status, pixel count, area percentage, and mean confidence.
6. **Detailed Scan Inspector:** Provided an interactive dropdown to select and inspect any scan in 4-column side-by-side format:
   - Original Ultrasound
   - SRAD + CLAHE Enhanced
   - Predicted Lesion Mask
   - Blended Overlay (Red lesion region + Yellow outer contour)
