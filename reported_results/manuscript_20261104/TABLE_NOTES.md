# Manuscript table captions and notes

Transcribed from the saved manuscript and supplementary tables on 4 November 2026.
CSV values retain the displayed precision; percentage-point differences may use unrounded evaluation values.

## table_1.csv

Table 1 Composition of the selected data sources. Split columns report image counts.

## table_2.csv

Table 2 Cross-model mask AP comparison on AgriParcel-MS. Values are percentages and rows are ordered by AP50–95. All six models use the training and test splits and common training budget in Section 2.6.1. Test results cover 2,000 images; AP summarizes mask overlap across confidence thresholds.

## table_3.csv

Table 3 Fixed-threshold detection, count, and area results at confidence 0.25. Detection and area-relative-error metrics are percentages. Small parcels have area ≤ 1,024 pixels in the 640 × 640 evaluation grid. Count MAE covers all images; matched-parcel MAPE excludes unmatched instances. WAPE and net bias use foreground-union areas, and net bias retains its sign. Bold indicates better performance: higher detection metrics, lower error metrics, and smaller absolute net area bias.

## table_4.csv

Table 4 Module ablation and boundary structure. ✓ and × indicate enabled and disabled modules. A = BDS, B = BAPS, and C = RBQS; the all-disabled row is the baseline and the all-enabled row is FieldBCQ. Values are percentages except score–IoU ECE (0–1). P and R use the evaluator-selected best-F1 operating point; AP50 and AP50–95 denote mask AP across confidence thresholds. Boundary-F1, merge rate, oversegmentation rate, and ECE use confidence 0.25. Matching uses mask IoU ≥ 0.50 and a boundary tolerance of 2 pixels on the 640 × 640 grid. Native F1 and AP75 are provided in Table S1. Bold indicates the best value in each metric column.

## table_5.csv

Table 5 Same-checkpoint comparison of class-only and joint scoring. Both conditions use the complete model’s trained weights; class-only scoring disables inference fusion. The joint-scoring column therefore repeats the FieldBCQ condition of Table 3. AP is native; other metrics use confidence 0.25. Differences are calculated from unrounded results and expressed in percentage points except for ECE. Bold indicates better performance.

## table_S1.csv

Table S1. Complete native operating-point and mask AP results for the eight configurations.

Values are percentages. Native P/R/F1 use each configuration’s evaluator-selected best-F1 operating point. A = BDS, B = BAPS, and C = RBQS. Boundary and structural metrics are reported in main Table 4.

## table_S2.csv

Table S2. Source-group segmentation results at confidence 0.25.

F1 and Boundary-F1 (BF1) are percentages computed from pooled within-group counts. Images and instances refer to the fixed test set. All three sources occur in training and testing; these groups do not constitute source-held-out evaluation.

## table_S3.csv

Table S3. Nominal-resolution groups at confidence 0.25.

F1 values are percentages; differences are percentage points calculated from unrounded results. Group F1 pools TP, FP, and FN; overall F1 cannot be obtained by averaging group F1 values. Nominal resolution is coupled to source, region, and density. Total counts cover 2,000 images and 21,190 annotations.

## computational_measurements.csv

Section 3.3: FieldBCQ improves segmentation performance with modest additional network complexity. Parameter count increases from 62.051 M to 63.970 M and computation from 297.03 to 305.34 GFLOPs, corresponding to increases of 3.09% and 2.80%. Under the timing protocol in Section 2.6.2, raw forward latency changes from 9.953 to 10.311 ms, and prediction-pipeline latency from 14.275 to 14.354 ms.

Section 2.6.2: Complexity is reported as parameter count and GFLOPs, with GFLOPs equal to twice the MAC count for the unfused evaluation graph. Parameters include auxiliary training branches, whereas evaluation-mode computation excludes the training-only boundary predictor. Per-image latency is measured on an RTX 4090 using FP32, 640 × 640 inputs, batch size 1, 50 warm-up iterations, and 200 CUDA-synchronized timed iterations. Raw forward latency covers network inference. Prediction-pipeline latency additionally includes preprocessing, NMS, and mask generation; both measurements exclude disk reading.
