# Introduction
This repository contains the dataset and code utilized in our research on developing **HSM-TDF**, a hard sample mining-based tongue diagnosis framework for fatty liver disease severity classification. 
# Paper Title
Tongue Diagnosis Framework for Fatty Liver Disease Severity Classification Using Kolmogorov-Arnold Network
# Method
We propose a Hard Sample Mining-based Tongue Diagnosis Framework (HSM-TDF) for fatty liver disease severity classificationy.
# Released Dataset
The dataset, named Tongue-FLD, includes 5,717 samples: 3,690 with non-FLD, 1,512 with mild FLD, and 515 with moderate/severe FLD, resulting in an imbalance ratio of 7.17/2.94/1.00. Each sample includes a segmented tongue image, FLD severity annotation and eight physiological indicators. The physiological indicators include Gender, Age, Height, Waist circumference (WC), hip circumference (HC), Weight, systolic blood pressure (SBP), and diastolic blood pressure (DBP). The data was obtained from a cohort study that received ethical approval. The participants were residents of Fuqing City, Fujian Province, China, aged 35 to 75 years. For each participant, a facial image with the tongue extended was captured using "Four Diagnostic Devices," and basic physiological indicators were measured. Subsequently, participants underwent ultrasound examinations, and FLD severity was assessed according to the standard criteria established by the Fatty Liver Disease Study Group of the Chinese Liver Disease Association. 
# Usage
Python: 3.8.19

    pip install -r requirements.txt
    cd ./Tongue-FLD
    cat Tongue_Images.tar.gz.* > Tongue_Images.tar.gz
    tar xzf Tongue_Images.tar.gz
    python random_rotate_images.py
    python rotate_pre_train.py
    python main.py

# Cite this repository
If you find this code or dataset useful in your research, please consider citing us:  
@inproceedings{HSM-TDF,  
  title={Tongue Diagnosis Framework for Fatty Liver Disease Severity Classification Using Kolmogorov-Arnold Network},  
  link={https://github.com/MLDMXM2017/HSM-TDF}  
}  

# Reference
[https://github.com/KindXiaoming/pykan](https://github.com/KindXiaoming/pykan)  
[https://github.com/Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan)  
[https://github.com/AntonioTepsich/Convolutional-KANs](https://github.com/AntonioTepsich/Convolutional-KANs)
[https://github.com/chenmc1996/LNL-IS](https://github.com/chenmc1996/LNL-IS)

