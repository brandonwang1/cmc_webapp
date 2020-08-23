# Cardiomyocyte Nuclear Identification



> As described in: [Deep Learning Identifies Cardiomyocyte Nuclei in Murine Tissue with High Precision](https://www.biorxiv.org/content/10.1101/2020.01.10.900936v2.full.pdf)
>
> Made by Brandon Wang, Hesham Sadek and MAIA Labs


We generated an artificial intelligence (AI) deep learning model that uses image segmentation to predict cardiomyocyte nuclei in mouse heart sections without a specific cardiomyocyte nuclear label. This tool can annotate cardiomyocytes with high sensitivity and specificity (AUC 0.94) using only cardiomyocyte structural protein immunostaining and a global nuclear stain.

To use this model to identify cardiomyocyte nuclei in tissue sections, please follow the instructions below. The data that the model generates will include total number of nuclei in the images, total number of cardiomyocyte nuclei, and an AI-generated image in which the cardiomyocyte nuclei will be identified.

* 8 μm thick mouse heart sections
* 20x zoom image
* One channel with DAPI stain and one channel with troponin T (TnT) stain (tool validated with Thermo Scientific MS-295-P mouse Troponin T antibody)
* Jpeg or tiff files only. Upload them to the data directory or specify a directory of your choice.

```shell
# 1. First, download/clone the repo
$ git clone https://github.com/brandonwang1/cmc_webapp.git
$ cd cmc_webapp

# 2. Install Python packages
$ pip install -r requirements.txt

# 3. Run!
$ python app.py
```

Open http://localhost:5000 to see the web interface.

Built with: Python, Tensorflow, Flask
