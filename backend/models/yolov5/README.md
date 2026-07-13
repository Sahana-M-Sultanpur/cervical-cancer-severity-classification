# YOLOv5 Weights

Place trained YOLOv5 cervical cell detection weights here or in:

```text
backend/saved_models/yolov5_cells.pt
```

The backend uses `backend/saved_models/yolov5_cells.pt` by default.

If the file does not exist, the prediction pipeline still runs by treating the
whole uploaded Pap smear image as one detected region. This fallback is useful
for testing the classifier before detection weights are available.
