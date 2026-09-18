Dataset
 ↓
Preprocessing (preprocess_data.py)
 ↓
Windowing (30s windows)
 ↓
SSL augmentation (physio_clr.py: SpectralSubbandMasking)
 ↓
Encoder (model.py: STFTEncoder2D, EEGEncoder)
 ↓
Projection head (model.py: SimCLR)
 ↓
SSL objective (temporal_loss.py, physio_clr.py)
 ↓
Checkpoint
(stft_pretrained_encoder.pth: STFTEncoder2D)
(simclr_encoder.pth: STFTEncoder2D/EEGEncoder)
(iclr_pretrained_encoder.pth: STFTEncoder2D/EEGEncoder)
(best_downstream_model.pth: STFTEncoder2D downstream)
(clinical_finetuned_model.pth: downstream finetuned)
(mil_production_model.pth: MIL downstream)
(mil_finetuned_model.pth: MIL downstream)
(mil_checkpoint.pth: MIL checkpoint)
 ↓
Downstream model (downstream_finetune.py, clinical_finetune.py, mil_model.py)
 ↓
Patient-level aggregation (downstream_finetune.py: evaluate)
 ↓
Evaluation (cross_cohort_eval.py, evaluate_roc.py, bootstrap_eval.py)
 ↓
Deployment (edge_benchmark.py, sleep_apnea_fp32.onnx, sleep_apnea_int8.onnx)
