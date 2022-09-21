# EFI Toolkit

With the widespread use of machine learning to support decision-making, it is increasingly important to verify and understand the reasons why a particular output is produced. Although post-training feature importance approaches assist this interpretation, there is an overall lack of consensus regarding how feature importance should be quantified, making explanations of model predictions unreliable. In addition, many of these explanations depend on the specific machine learning approach employed and on the subset of data used when calculating feature importance.

A possible solution to improve the reliability and understandability of explanations is to combine results from multiple feature importance quantifiers from different machine learning approaches coupled with data re-sampling. 

EFI toolkit implements this solution using:
- State-of-the-art information fusion techniques
- Fuzzy sets

The toolkit provides complete automation of the entire feature importance computation cycle. 

### INPUT = Structured data


## The main attributes of the toolbox are: 
- automatic training and optimisation of machine learning algorithms.
- automatic computation of a set of feature importance coefficients from ensemble of optimised machine learning algorithms and feature importance calculation techniques.
- automatic aggregation of importance coefficients using multiple decision fusion techniques.
- automatic generation of fuzzy membership functions that show the importance of each feature to the prediction task in terms of `low', `moderate' and `high' importance as well as their levels of uncertainty.

![alt text](https://github.com/jimmafeni/EFI-Toolbox/blob/main/featureimportance.png)

![alt text](https://github.com/jimmafeni/EFI-Toolbox/blob/main/fefitoolkit.PNG)

## Relevant publications

If you use auto-sklearn in scientific publications, we would appreciate citations.

**Efficient and Robust Automated Machine Learning**
*Matthias Feurer, Aaron Klein, Katharina Eggensperger, Jost Springenberg, Manuel Blum and Frank Hutter*
Advances in Neural Information Processing Systems 28 (2015)

[Link](https://papers.neurips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf) to publication.
```
@inproceedings{feurer-neurips15a,
    title     = {Efficient and Robust Automated Machine Learning},
    author    = {Feurer, Matthias and Klein, Aaron and Eggensperger, Katharina and Springenberg, Jost and Blum, Manuel and Hutter, Frank},
    booktitle = {Advances in Neural Information Processing Systems 28 (2015)},
    pages     = {2962--2970},
    year      = {2015}
}
```

----------------------------------------

**Auto-Sklearn 2.0: The Next Generation**
*Matthias Feurer, Katharina Eggensperger, Stefan Falkner, Marius Lindauer and Frank Hutter**
arXiv:2007.04074 [cs.LG], 2020

[Link](https://arxiv.org/abs/2007.04074) to publication.
```
@article{feurer-arxiv20a,
    title     = {Auto-Sklearn 2.0: Hands-free AutoML via Meta-Learning},
    author    = {Feurer, Matthias and Eggensperger, Katharina and Falkner, Stefan and Lindauer, Marius and Hutter, Frank},
    booktitle = {arXiv:2007.04074 [cs.LG]},
    year      = {2020}
}


### The toolkit and its description will be updated as new explainability techniques and machine learning models are implemented.
