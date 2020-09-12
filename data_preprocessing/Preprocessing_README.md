# Steps to run data prprocessing

- Step 1: convert_to_nii.py
- Step 2: acpc_realignment.r (we can also use spm12 matlab, I used R as it provided convinient wrapper)
  ### Images after after realignment.
![im6](https://github.com/shashank3110/Master_Thesis_BA_DeepVis/blob/master/static_files/im6.png)
- Step 3: segment_oasis.m
  ###  Gray matter and White matter segments.
![im7](https://github.com/shashank3110/Master_Thesis_BA_DeepVis/blob/master/static_files/im7.png) 
![im8](https://github.com/shashank3110/Master_Thesis_BA_DeepVis/blob/master/static_files/im8.png) 
- Step 4: generate_dartel_template.m
  ###  Dartel Template for Gray matter segment
![im9](https://github.com/shashank3110/Master_Thesis_BA_DeepVis/blob/master/static_files/im9.png)
- Step 5: run_dartel.m
  ###  Preprocessed Gray matter segment.
![im5](https://github.com/shashank3110/Master_Thesis_BA_DeepVis/blob/master/static_files/im5.png)
- Step 6: tfrecords_conversion/convert_to_tfrecord.py
