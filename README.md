# AudioMask

'AudioMask: Robust Sound Event Detection Using Mask R-CNN and Frame-Level Classifier' https://ieeexplore.ieee.org/document/8995448

### Data Acquiring Process:
	1. Create the desired dataset with desired event probability from TUT dataset or use one of the already created datasets
	2. Create mel-spectrograms of audio files with their masks for Mask R-CNN
	
### Mask R-CNN:
	1. Either train Mask R-CNN model for the specific event using a training set of the event or use the already trained one
	2. Run Mask R-CNN on the test set and produce all of the regions with event presence probability above 0.5

(Using process_file_for_evaluation.py) Read the Mask R-CNN report and sort it

### Frame-level Classifier:
	1. Either train frame-level classifier with segments of the data from the audio files or use the trained ones
	2. Convert list of regions proposed by Mask R-CNN to segments
	3. Run frame-level classifier on these segments
	4. Choose the true segments based on the probability produced by the classifier and confidence of the Mask R-CNN

Calculate the F1-score and ER
