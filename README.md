# Nb2Nb
This project aims to understand the idea of the paper ,"Neighbor2Neighbor:Self-Supervised Denoising From Single Noisy Images". Since this code is an unofficial implementation, some details may differ from the original description in the paper. To make it easier to understand the underlying theory, All the codes are written by Python and Tensorflow.

## Sample Result
All the results are tested on [SIDD Validation Dataset](https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php)

Noisy Image|denoised Result
----|-----
![21](https://github.com/DavidQiuChao/Nb2Nb/blob/main/figs/0021_noisy.png)|![21c](https://github.com/DavidQiuChao/Nb2Nb/blob/main/figs/0021_clean.png)
![37](https://github.com/DavidQiuChao/Nb2Nb/blob/main/figs/0037_noisy.png)|![37c](https://github.com/DavidQiuChao/Nb2Nb/blob/main/figs/0037_clean.png)
![56](https://github.com/DavidQiuChao/Nb2Nb/blob/main/figs/0056_noisy.png)|![56c](https://github.com/DavidQiuChao/Nb2Nb/blob/main/figs/0056_clean.png)

## Update
Testing files, including trained model, are uploaded. The main testing file is "test.py", which can easily run by input the command 'python test.py -s saves -n nets.Unet -d "dataDir" -r "resultDir"'. The "dataDir" specifies the testing data directory, and the "resultDir" is the path for saving result. 

For rendering the '.mat' data, using the [Simple Camera Pipline](https://github.com/AbdoKamel/simple-camera-pipeline).

More training files will be uploaded as soon as possible. To be continue ...

