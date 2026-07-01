# DUPLICATE IMAGE IDENTIFIER

### Definition:

    This project is designed to identify duplicate images in a given dataset. It uses image hashing techniques to compare images and determine if they are duplicates. With score logic, duplication sensibility can be adjusted to suit different use cases. Every dataset has unique image calibrations, so precision of the duplication must be considered.

### Usage:

```bash
python detectDuplicates.py
```

Detects duplicate images in the specified dataset directory, depending on precision. There are 3 scenarios for duplication detection:

- **Vanilla**: Use the techniques as in priority: IoU -> P hashing -> CLIP embedding (The most appropriate prioritization). This technique is the <u>most fastest</u>, but <u>least sensitive</u>. All of techniques considered seperately, based on their threshold values. Must been careful when setting threshold values of each technique, the image isn't duplicated when all thresholds fallen.

- **Scoring**: With activating it (activateScore = True), duplication sensibility declared by all of 3 techniques: IoU, P hashing and ClIP embedding. Each technique has its own threshold, and effect general scoring with their score ration values (Default: IoU = 0.5, P hashing = 0.3, CLIP embedding = 0.2). If the general total score is higher then the score threshold (Default: 0.7), the image is considered as duplicated. This technique is the <u>most controllable</u> technique. Playing with score/threshold values must be considered, based on the dataset.

- **Scoring with LPSIS (NOT DONE YET!)**: Scoring technique with adding LPSIS (Local Patch Similarity Index Score) technique. The main goal of this option is to improve the sensibility of duplication detection for some specific cases. In future adjustments, the LPSIS technique can be replaced with other sensitive technique or tool. Will be the <u>most sensitive</u> option.

After running the script, there will be dictionary output in the terminal (the duplication results if verbose == True, based on which option is chosen). The key index value of the value represent the actual image index in the dataset, and value is a list of the duplicated image indices, including key index value.
<br><br>
For example, if the output is {0: [0, 1, 2]}, it means that the image at index 0 is duplicated with images at indices 1 and 2.
<br><br>
Make lock = True if you want to save exact images from your dataset to a new directory (which should be specified with attachment to output_dir) in save_clusters() function.

### Displaying:

```bash
python displayImg.py
```

Firstly, the directory of the dataset must be entered correctly (dir_path = "{directory_path_images_will_be_displayed}"). Then, an input option will show up in the terminal:

```bash
Enter image indices (comma-separated):
```

- Enter the indices of the images you want to display, seperated by commas. 
<br>
For example; 1, 3,4, 5, 6
<br>
- With pressing right/left arrow keys, you can navigate through the images. For quiting the display, press the 'esc' key. Also automatically quits after the last image is passed. It is much easier to use after "detectDuplicates.py" is run.

### Results & Conclusion:

The results are satisfying for some datasets, but for more sensitiveness required datasets although, the 3rd option must be done. Although the 3rd option is not done yet, an interpretation can be done as identified duplicates with most aggressive 2nd option is showing that the kinda similar images are not adding any uniqueness/effectiveness to remember the path to classify the exact disaease type.