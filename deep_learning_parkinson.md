# Deep Learning based classification of Parkinson and other diseases from EMG data 

Many methods for images [1](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9139649/)
A review on the subject will be [this] (Deep learning for neurodegenerative disorder (2016 to 2022): A systematic review). his paper documents the systematic reviews on the detection, and classification techniques of neurodegenerative disorder from five different facets viz., datasets and data modality of neurodegenerative disorder, pre-processing methods, deep learning-based detection and classification of neurodegenerative disorder, and performance measure matrices. However, the full text is not freely available.

### [Classification of Parkinson’s Disease EEG Signals Using CNN and ResNeT](https://www.researchgate.net/publication/355483890_Classification_of_Parkinson's_Disease_EEG_Signals_Using_CNN_and_ResNet)

 The EEG signals have 64 channel and were collected by a BrainVision system with the sampling rate of 500 Hz
After the BSS phase, the data was segmented i nto 1000 sampl e segments where each segment represent 2s of data. At this point the data is ready for the classification phase.  

Conclusion:

The paper lacks details and it will be hard to reproduce.


### [Automatic Resting Tremor Assessment in Parkinson’s Disease Using Smartwatches and Multitask Convolutional Neural Networks] (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7794726/)

[Parkinson’s Disease Detection from Drawing Movements Using Convolutional Neural Networks](https://www.mdpi.com/2079-9292/8/8/907)
This paper contributes to this effort by analyzing a convolutional neural network (CNN) for PD detection from drawing movements.
The inputs to the CNN are the module of the Fast Fourier’s transform in the range of frequencies between 0 Hz and 25 Hz. 
The sample sequence was divided into 3-second windows (330 samples per window) separated by 0.5 seconds (it means a 2.5-second overlap between two consecutive windows). All the windows from PD subjects were labeled as class 1 and all 3-second windows from healthy subjects were labeled with class 0. Each window was expanded to 512 points using zero padding. After that, the module of the Fast Fourier’s Transform (FFT) was obtained using a Hamming windowing. 

Conclusion: 
Image-based CNN.

### [Parkinson's Disease Detection using Convolutional Neural Networks] (https://ejmcm.com/article_3671_023b1a587f7bd67e45915ea208e2def7.pdf)
This CNN comprises two parts: extraction and arranging
(completely linked layers). CNN involves two pieces. CNN refers to the increase in frequency volume from 0 Hz
to 25 Hz by the Fast Fourier Module For Fast Fourier's Transform (FFT), a
hamming window was used. Since the FFT is symmetrical for real signals, a 256-point spectrum
representation was produced in the 0-55 Hz recurrence. From this image, the initial 125 goals of
the chosen range for the recurrence band 0–25 Hz were deemed negligible because of the vitality
of the recurrence spectrum over 25 Hz less than 1% of the overall vitality. Figure 2 says of all the  

### [A hybrid deep transfer learning-based approach for Parkinson's disease classification in surface electromyography signals] (https://www.sciencedirect.com/science/article/abs/pii/S1746809421007588?casa_token=QbD_xFur8RkAAAAA:XBF_bTamOHOxq5Qv-mU_ffU16jAMbIcbw8Otc89WFPGAU7osZQZozZe-3tdCVtfoDxhWsXNpKg)

Surface Electromyography (sEMG) signal trials received from the upper extremities, such as the arm and wrist, would be an efficient way to assess neuromuscular function in the detection of PD. his paper mainly aimed to utilize pre-trained deep transfer learning (DTL) structures and conventional machine learning (ML) models as an automated approach to diagnose PD from sEMG signals. Primarily, we stacked the extracted features from three deep pre-trained architectures, including AlexNet, VGG-f, and CaffeNet, to generate the discriminative feature vectors. Although the number of stacked features from all the three deep structures was large, the proper features is effective in overcoming the challenge of over-fitting as well as increasing the robustness to added noise with different levels. . Finally, we utilized the support vector machine (SVM) with radial basis function (RBF) kernel for identifying PD disorder. The experimental results in different analysis frameworks illustrated that the hybrid deep transfer learning-based approach to PD classification could lead to hitting rates higher than 99%. n this regard, our method provides evidence of an automated procedure that analyzes sEMG signals, based on which it extracts features using a stacking of three DTL networks. 
Spectrograms as an input.sEMG spectrogram images were acquired every 500 ms with 30–50% overlapping from recorded signals (i. e., windowing procedure). A total of 24,000 spectrogram frames of subjects were built. Using the labeling process, three different convolutional DNN models were trained individually.

Conclusion: 

Image-based method with classical CNNs from spectrograms.  They stacked all possible extracted features from three deep pre-trained architectures, including AlexNet, VGG-f, and CaffeNet, to generate the discriminative feature vectors. 
good review of other methods for EMG based Parkinson classification.

### [Deep Learning with Convolutional Neural Networks Applied to Electromyography Data: A Resource for the Classification of Movements for Prosthetic Hands] (https://www.researchgate.net/publication/307919884_Deep_Learning_with_Convolutional_Neural_Networks_Applied_to_Electromyography_Data_A_Resource_for_the_Classification_of_Movements_for_Prosthetic_Hands)

Conclusion:
 
Old but one of the few which uses non-images.