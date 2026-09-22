# Image processing

There are four channels Bright Field (BF), DAPI (Blue), GFP (Green), RFP (Red). The bright filed is just for non-fluorescence imaging. All the other three are fluorescence image. Each color means a parcitular strain. 

In this tif image, there are many layers representing z-stacks. So each image shows the 3D distribution of species. 

## Reproducible raw-processing inputs

The original image-processing notebook export is [RawImageProcessing.py](RawImageProcessing.py). It remains unchanged as the record of the original processing steps. The reviewer-facing commands run the same core processing against one explicitly supplied data-download directory. They discover the published filename convention directly, so the reviewer does not create a manifest or edit source paths. Experiment 1.3.2's `Activity_1.3_rep7` has acquisition folders directly at its root, while Experiment 1.3.1's `Activity_1.3_rep1`–`rep3` put acquisition folders below `Chitin`, `Chito5`, or `NAG`; both layouts are supported and a mixed layout is rejected.

For each BF stitched mask named `<sample>_<substrate>_1_w1SD BF_stitched_mask.tif`, the command expects the three fluorescence channels in the same directory, each with eight tiles named `<sample>_<substrate>_1_w[2|3|4]SD <channel>_s[1..8].TIF`. This is the published layout used by `RawImageProcessing.py`.
