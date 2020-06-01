A new deep learning baseline for image classification
### Quick run tutorial:   
1.Input two parameters to run, the first parameter “--config_file” specifies the path of file, the second “--run_type” specifies which test to run. 
2.The results will be in the folder “PCANet1_results”.
  
- --config_file: the path of configuration file  
- --run_type: specify which test to run  

- --run_type d # run on demo dataset 
- --run_type t # run on tiny dataset  
- --run_type f # run on full dataset  

* Sample command: 
python experiment_PCANet1_main.py --config_files PCANet1_configs\demo.json --run_type d 


### Script
#####	experiment_PCANet1_main.py #representations of Fourier,Wavelet,.....
#####	experiment_PCANet2_main.py #representation of 2d-Fourier on different datasets  


### Example
#### Command:  
- python experiment_PCANet1_main.py --config_files PCANet1_configs\demo.json --run_type d  
#### Result(basic-demo):  
X.shape: (49, 48400)  
filter.shape: (2, 7, 7)  
I_layer1.shape: (100, 2, 28, 28)  
int_img_list.shape: (100, 28, 28)  
I_layer1.shape: (20, 2, 28, 28)  
int_img_list.shape: (20, 28, 28)  
train_feats.shape: (100, 256), test_feats.shape: (20, 256)  
PCANet  
l: 2  
patch_size: (7, 7)  
stride: 1  
block_size: (7, 7)  
block_stride: 3  
method: Fourier  
stage: 1  
reduction_method: exponent  
feature_method: histogram  
fourier_basis_sel: magnitude  
Mean Accuracy Score: 0.85  



