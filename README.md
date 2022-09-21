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

![alt text](https://github.com/jimmafeni/EFI-Toolbox/blob/main/fefitoolkit.png)


## The toolkit and its description will be updated as new explainability techniques and machine learning models are implemented.
