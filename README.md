# NeSyALC

This repository provides the code for training NeSyALC, a neural-symbolic model that can use the knowledge with expressivity no more than $\mathcal{ALC}$ from any OWL ontologies to guide the learning of neural models.

## Overview

The input of NeSyALC is an OWL ontology and a neural model. In the structure of NeSyALC, the domain of discourse is the object set of the neural model, and the signature is the union of concept names and role names in the ontology. By reconstructing the output of the neural model into an interpretation of the ontology, we can revise the neural model through learning with the hierarchical loss presented in the paper. The parameters of the neural model can also be revised in a way of multi-task learning, which is easy to be extended with the codes in this repository.
The output of NeSyALC is the revised neural model output (/revised neural model parameters with the extension).

## Details for Reproducing

### Preprocessing for the OWL ontology

The input OWL ontology should be truncated into its $\mathcal{ALC}$ fragment, and then be normalized. 
Run the following command with **JDK 1.8** under [the root of this directory](https://github.com/AnonymousResearcherOpen/NeSyALC/): 

    java -jar Normalization.jar training/ontologies training/input

The output of preprocessing is the files in 'training/input':

- 'concepts.txt', 'roles.txt', 'individuals.txt': the concept names(/role names/individual names) set.
- 'normalization.txt': the nomalized TBox axioms.
- 'abox.txt': the abox assertions.
- 'subclassaixoms.txt': the original TBox axioms.

Note: The source code of 'Normalization.jar' and 'CQGenerator.jar' is in [normalization](https://github.com/AnonymousResearcherOpen/NeSyALC/tree/main/normalization). If you want to repackage the jar based on our source code, remember to delete all dependencies named 'owlapi-xxx.jar' in the artifact, while only remain the 'owlapi-distribution-5.1.3.jar'. 

### Training

The training and evaluation is in [training](https://github.com/AnonymousResearcherOpen/NeSyALC/tree/main/training), to train NeSyALC, run:

    python .\run.py --info_path input --out_path output --save_path output --iter_path ontologies --mask_rate 0.2 --alpha 0.8 --device_name cpu --model_name Godel/Rule

For evaluation, we randomly masked the ABox of the input ontology as the initial output of the neural models, so can evaluate the performance of NeSyALC when meeting with different distributions. The generation of the masked ABox (imitation of the output of a neural model) is in [Evaluation.MaskABox](https://github.com/AnonymousResearcherOpen/NeSyALC/tree/main/training/Evaluation.py), the masked ABox and the original ABox are saved in '--save_path'. And the mask rate is designated by '--mask_rate'. While '--alpha' is the threshold of truth value for the transformation between fuzzy ALC and crisp ALC. And the masked value is in the range of (1-alpha, alpha). The model also supports using GPU, with '--device_name cuda:0'.

For comparison with the baselines (LTN, BoxEL, Box2EL, Falcon, ELEmbedding), run:

    python .\run.py --info_path input --out_path output --save_path output --iter_path ontologies --mask_rate 0.2 --alpha 0.8 --device_name cpu --model_name baseline_name

The settings of the training parameters are: 

    Learning rate with the Adam optimizer -- 2e-4
    Threshold for fuzzy-to-crisp transformation -- 0.8
    epoch size -- 50000
    batch size -- 64
    Early stopping criteria -- patience=10 epochs with 1e-4 improvement threshold

### Performance Evaluation

The revised results of NeSyALC and baselines are evaluated under the semantics of fuzzy first-order logic, with codes in [training/evaluation](https://github.com/AnonymousResearcherOpen/NeSyALC/tree/main/training/evaluation/). To compute the successful rate, run [run.ipynb](https://github.com/AnonymousResearcherOpen/NeSyALC/tree/main/training/evaluation/run.ipynb)

To do the conjunctive query answering (CQA) evaluation, firstly,
generate the conjunctive queries and answers:

    java -jar CQGenerator.jar training/ontologies training/input

Then run [CQAnswering_evaluation.ipynb](https://github.com/AnonymousResearcherOpen/NeSyALC/tree/main/training/CQAnswering_evaluation.ipynb) to generate the CQA evaluation results.

### Application: Semantic Image Interpretation
ALC ontologies and EL ontologies are saved in [SII/ontologies](https://github.com/AnonymousResearcherOpen/NeSyALC/tree/main/SII/ontologies).
Run evaluation with NeSyALC on notebook [SII_code/dfalc_run.ipynb](https://github.com/AnonymousResearcherOpen/NeSyALC/tree/main/SII/SII_code/dfalc_run.ipynb). Run evaluation with LTN on notebook [SII_code/run_ltn.ipynb](https://github.com/AnonymousResearcherOpen/NeSyALC/tree/main/SII/SII_code/run_ltn.ipynb). 

## Dependencies

    JDK 1.8
    python 3.7.0
    torch 1.8.1
    python-csv 0.0.13
    matplotlib 3.3.2
    pickle 4.0
    numpy 1.21.4
    pandas 1.1.3
    pyparsing 3.0.6
    loguru 0.6.0

## Performance Evaluation Results

Results of NeSyALC and baselines are output in [output](https://github.com/AnonymousResearcherOpen/NeSyALC/tree/main/training/output/), [product_output](https://github.com/AnonymousResearcherOpen/NeSyALC/tree/main/training/product_output/), respectively. We zipped the training results in [results](https://drive.google.com/drive/folders/1ob0RVM6GwAQvgew9yZTrCfNrfvbWFKRb?usp=sharing).


