[//]: # ([![License]&#40;https://img.shields.io/badge/license-MIT-blue.svg&#41;]&#40;LICENSE&#41;)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Repository Size](https://img.shields.io/github/repo-size/marta-seq/PENGUIN.svg)](https://github.com/marta-seq/PENGUIN)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)


# Machine and deep learning to predict Viral fusion peptides 


This repository contains the files for the paper : 


## Table of Contents
- [Project Description/Abstract](#project-description)
- [Main contributions of this work](#main-contributions-of-this-work)
- [Credits](#credits)
- [License](#license)
- [Contributing](#contributing)

## Project Description/Abstract

Viral fusion proteins located on the surface of enveloped viruses like SARS-CoV-2, Influenza, and HIV, play a vital role in fusing the virus with host cell membranes. Fusion peptides, conserved segments within these proteins, are crucial for the fusion process and are potential targets for therapy. Experimental identification of fusion peptides is time-consuming and costly, making bioinformatics tools essential. Although homology methods have been used towards this end, they fail to identify fusion peptides lacking overall sequence similarity to known counterparts. Therefore, alternative methods that do not rely on sequence identity are needed to discover new putative fusion peptides. Recent advances in artificial intelligence, particularly in protein classification, have confirmed that machine learning (ML) methods can develop promising approaches.
A scheme overview of Viral Fusion Protein: 

![plot](figs/vfp_image.png)

In this study, we explore various ML-based approaches to identify fusion peptides within a fusion protein sequence. We employ token classification methods and sliding window approaches coupled with machine and deep learning models. We evaluate different protein sequence representations, including one-hot encoding, physicochemical features, as well as representations from Natural Language Processing, such as word embeddings and transformers. Through the examination of over 50 combinations of models and features, we achieve promising results. The most successful models achieved are based on a state-of-the-art transformer for aminoacid token classification. Furthermore, we utilize the best models to predict hypothetical fusion peptides from SARS-CoV-2, and critically analyse annotated peptides from existing research. 
Overall, our models effectively predict the location of fusion peptides, even in viruses for which limited experimental data is available.



General view of the project: 
![plot](figs/projectscheme.png)

## Main contributions of this work:
- A curated dataset of Viral Fusion porteins and fusion peptides
- We explore diverse techniques to detect fusion peptides within fusion protein sequences, employing strategic approaches and protein representations. 
- Our models demonstrate high performance in accurately identifying fusion peptides within viral fusion proteins. 
- Notably, physicochemical sliding window approaches and transformer-based token classification models exhibit remarkable performance, with the latter showing superior results. 
- We  provided biological insightful observations on annotated fusion peptides (check paper).
- These results emphasize the potential for improvement, particularly when a more complete and updated dataset is made available.


## Credits
If you find this repository useful in your research or for educational purposes please refer to:


## License

Developed at:
1. Department of Informatics, School of Engineering, University of Minho, Braga, Portugal 
2. ITQB NOVA, Instituto de Tecnologia Química e Biológica António Xavier, Universidade Nova de Lisboa, Oeiras, Portugal


Released under the GNU Public License (version 3.0).


[//]: # (.. |License| image:: https://img.shields.io/badge/license-GPL%20v3.0-blue.svg)
