#Steps to run data prprocessing

- Step 1: convert_to_nii.py
- Step 2: acpc_realignment.r (we can also use spm12 matlab, I used R as it provided convinient wrapper)
- Step 3: segment_oasis.m
- Step 4: generate_dartel_template.m
- Step 5: run_dartel.m
- Step 6: tfrecords_conversion/convert_to_tfrecord.py
