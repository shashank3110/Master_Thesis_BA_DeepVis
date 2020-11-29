# Master Thesis: Deep Visualization for MR based Biological Age Estimation.
### Master thesis code version control.

- Data Preprocessing refer:  data_preprocessing/..   <br/>  (refer the [Preprocessing_README.md](https://github.com/shashank3110/Master_Thesis_BA_DeepVis/blob/master/data_preprocessing/Preprocessing_README.md) )

- CA Estimation and BA Estimation training code (had to be made private due to copyrighted data involved.)
  
- Visualization code  colab notebooks refer: colab_notebooks/.. :
  - [colab_notebooks/ba_estimation_network_saliency_maps_gcam_gcam++_notebook.ipynb](https://github.com/shashank3110/Master_Thesis_BA_DeepVis/blob/master/colab_notebooks/ba_estimation_network_saliency_maps_gcam_gcam%2B%2B_notebook.ipynb)
  - [colab_notebooks/ba_estimation_network_saliency_maps_gcam++_notebook_save_intermediate_maps.ipynb](https://github.com/shashank3110/Master_Thesis_BA_DeepVis/blob/master/colab_notebooks/ba_estimation_network_saliency_maps_gcam++_notebook_save_intermediate_maps.ipynb) (same as the above notebook but it saves intermediate maps along with the final result.)
  - [colab_notebooks/siamese_network_saliency_maps_gcam_gcam++_notebook.ipynb](https://github.com/shashank3110/Master_Thesis_BA_DeepVis/blob/master/colab_notebooks/siamese_network_saliency_maps_gcam_gcam%2B%2B_notebook.ipynb)

### Visualization Generated: Activated Regions on applying SMOE Saliency Maps [1] with GRADCAM++ [3]
![cover](https://github.com/shashank3110/Master_Thesis_BA_DeepVis/blob/master/static_files/healthy_activated_regions.png)

### References
- [1] [Efficient Saliency Maps for Explainable AI](https://arxiv.org/abs/1911.11293)
- [2] [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
- [3] [Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks](https://arxiv.org/abs/1710.11063)
- [4] [Smooth Grad-CAM++: An Enhanced Inference Level Visualization Technique for Deep Convolutional Neural Network Models](https://arxiv.org/abs/1908.01224)

### Brain Atlas for reference (requires flash player support enabled on browser):
- http://www.thehumanbrain.info/head_brain/horizontal.php
- http://www.thehumanbrain.info/head_brain/hn_coronal_atlas/coronal.html
- http://www.thehumanbrain.info/head_brain/hn_sagittal_atlas/sagittal.html
