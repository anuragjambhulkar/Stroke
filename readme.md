```

---

## requirements.txt

```text
gradio>=3.0
opencv-python>=4.5
mediapipe>=0.10
numpy>=1.22
scipy>=1.9
```

---

## README.md

````md
# Unilateral Facial Paralysis Simulator

A local Gradio Python app that:

- uploads a frontal face image
- detects MediaPipe Face Mesh landmarks (refined landmarks enabled)
- applies localized piecewise affine warps across Delaunay-triangulated lip triangles to simulate unilateral facial droop
- exposes sliders for severity and lateral pull and a side selector

## Recent Changes

* **Improved Beard Handling:** The simulation now gracefully handles faces with beards by excluding the chin and jawline from the warping process. This results in a more natural and realistic simulation for users with facial hair.
* **Updated Slider Labels:** The slider labels in the Gradio interface have been updated to be more descriptive. "Severity" is now "Droop Intensity," and "Lateral Pull" is now "Mouth Stretch." This should make the tool more intuitive and user-friendly.

## Setup

1. Create a Python virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
````

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

Visit [http://localhost:7860](http://localhost:7860) in your browser.

## Tips

* Use frontal images where the face is mostly visible and not heavily rotated. MediaPipe works best with good lighting.
* Glasses: thin-framed glasses are usually fine; very thick or reflective glasses may cause landmark dropouts.
* Low light: increase illumination. Landmark detection may fail if the face is not detected.

## Known limitations

* Landmark index choices for the lip region are commonly-used but may require tuning for particular faces. I intentionally kept the lip index set conservative so the effect is localized; you can expand `LIPS_OUTER` and `LIPS_INNER` in `app.py`.
* The app warps only a local triangulated region — this keeps deformation localized and more natural but sometimes requires adding cheek/eyebrow anchors for stronger asymmetry.
* Extreme slider values may create unnatural stretching or fold artifacts. Use moderate values (0.3–0.7) for realistic results.

## Next steps

* Finalize the lip region indices for your dataset, and add eyebrow/cheek triangles if more asymmetry is desired.
* Add an option to preview detected landmarks and triangle overlays (useful for tuning indices).
* Save output images and create side-by-side comparison UI.

---

If you want, I can now:

* adjust the lip indices to a different canonical set,
* add an overlay debug mode that draws landmarks + triangles,
* or convert this into a small Flask app that serves the same processing behind an API.
