In the OCR pipeline, two experiments are computed:
(1) No preprocessing
(2) After preprocessing & postprocessing

FOLDERS:
---------------------------------------
1. Input_Images_And_Annotations
Contains the original images and the annotations (annotated by myself) used to run the experiments.

2. EasyOCR_Output_NoPreprocessing
Output images produced after running the experiment WITHOUT preprocessing.

3. EasyOCR_Output_WithPreprocessingAndPostprocessing
Output images produced after running the experiment WITH preprocessing and postprocessing.

4. Results
Contains the experiment results stored in CSV format.


SCRIPTS:
----------------------------------------
1. EasyOCR_No_Preprocessing.py
Runs experiment (1) - OCR without any preprocessing.

2. EasyOCR_With_Preprocessing_And_Postprocessing.py
Runs experiment (2) - OCR with preprocessing and postprocessing applied.
