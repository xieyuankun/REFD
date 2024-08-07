# REFD
 This repository presents our work titled "Generalized Source Tracing: Detecting Novel Audio Deepfake Algorithm with
Real Emphasis and Fake Dispersion Strategy," which was available on arxiv at "https://arxiv.org/abs/2406.03240".


### The code for this project is currently unorganized and somewhat disordered. Further updates are expected.

## ADD2023 Track3 dataset
Track 3 Training/Development Dataset: https://zenodo.org/records/12179632

Track 3 Evaluation Dataset: https://zenodo.org/records/12179884

## Code

## Real Emphasis (RE) Stage

### 1. Offline Feature Extraction
```
cd code/ADD2023t3_RE
python preprocess_da_full.py 
```

### 2. Training
```
python main_train.py
```

### 3. Test 
Generate RE score in result/result_RE.
```
python generate_score_da.py
```

## Fake Dispersion (FD) Stage

### 1. Training
```
cd code/ADD2023t3_FD
python main_train.py
```

### 2. Preprocessing for OOD Detection
Training and testing features are saved in `./ood_step/traindict.pt` and `./ood_step/evaldict.pt`, respectively, for subsequent OOD score calculations.

FD test scores are stored in `/result/result_FD.txt`.

```
python ood_detector_pre.py
```

### 3. OOD Score Calculation
Select a score-based OOD detector in the arg ¡°ood_detector_name¡± parameter for OOD score calculation. 

OOD scores are saved in `./ood_step/result_oodscore.pt`.

```
python ood_detector.py
```

### 4. Final F1 Score Calculation
Please adjust line 17 to select the appropriate threshold for determining the OOD class. For other details, refer to the source code and comments.

```
python generate_f1_e2e.py
```