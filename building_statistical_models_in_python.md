---
lang: eng
annotation-target: 
author:
  - Huy Hoang Nguyen
  - Paul N Adams
  - Stuart J Miller
subject:
  - Statistical Anylisis
  - Python
tags:
  - book
  - statisticalmodels
---
## Outline

- An introduction to statistics 
- Regression models 
- Classification models 
- Time series models
- Survival analysis
## Readthrough

# Part 1, An introduction to statistics

## Chapter 1, Sampling and Generalization

#### Population versus sample

- The goal of stats modeling is to answer a question about a group by making an inference about that group (the entirety of the group is called a **population**) .
- Because it's unlikely to have data on the whole population (can't collect all data, too large),  we use a subset of the population, a **sample**.
- This subset needs to be **representative** of the population.


### Population inference from samples

- We have to give our study the same degrees of uncertainty as those of the population, to that effect we used **randomized experiments**.
- A random experiment has two characteristics: 
	- **random sampling**
	- **random assignment of treatments**

#### Random assignment of treatments

Refers to two things :
- gain an understanding of specific input variables and their influence 
on the response
- remove the impact of external variables on the outcomes of a study. These external variables are called **confounding variables**.
```
Referring back to the example in the earlier section, Population versus sample, consider a farmer who decides to start using pesticides on his crops and wants to test two different brands. The farmer knows there are three distinct areas of the land; plot A, plot B, and plot C. To determine the success of the pesticides and prevent damage to the crops, the farmer randomly chooses 60 plants from each plot (this is called stratified random sampling where random sampling is stratified across each plot) for testing. This selection is representative of the population of plants. From this selection, the farmer labels his plants (labeling doesn’t need to be random). For each plot, the farmer shuffles the labels into a bag, to randomize them, and begins selecting 30 plants. The first 30 plants get one of two treatments and the other 30 are given the other treatment. This is a random assignment of treatment. Assuming the three separate plots represent a distinct set of confounding variables on crop yield, the farmer will have enough information to obtain an inference about the crop yield for each pesticide brand.
```

### Observational study

- Researcher learns through observing data but cannot make cause-and-effect conclusion like with a randomized experiment (because it lacks one of the two citeria of it).
- Done when randomized experiment is not possible (data from random experiment already exists, too costly or just impossible)
- A way to derive causality from observational studies is through **repeated random sampling** and **repeated analysis**.

#### Statistical test 
- Statistics is to help make decisions based on quantifiable uncertainties
- A hypothesis test contains a **null hypothesis** (no difference between data) and an **alternative hypothesis** (difference between data), difference based on a **critical value**, a benchmark
- A hypothesis test can test the following:
	- One variable against another (such as in a t-test)
	- Multiple variables against one variable (for example, linear regression)
	- Multiple variables against multiple variables (for example, MANOVA)

### Sampling strategies

Two types of sampling methods:

#### Probability sampling

- A sample is chosen based on a theory of probability, or randomly with random selection (every member has the same chance)

4 types of probability sampling.

- **Simple random sampling** : every member has an equal chance. Used when all member have similar properties related to important features. Minimizes bias and maximizes representativeness.
- **Systematic sampling** :  based on an fixed interval, choose a random numbered data point and select the rest of the data along the interval. Less random than simple random sampling. With systematic sampling, there is a biased risk when the list of members of a population is organized to match the sampling interval.
```
In a class at one high school in Dallas, there are 50 students but only 10 books to give to these students. The sampling interval is fixed by dividing the number of students in the class by the number of books (50/10 = 5). We also need to generate a random number between one and 50 as a random starting point. For example, take the number 18. Hence, the 10 students selected to get the books will be as follows: 18, 23, 28, 33, 38, 43, 48, 3, 8, 13
```

- **Stratified sampling** : based dividing a population a subpopulations called **strata**. These subpopulations must be distinct so that every member in each stratum has an equal chance of being selected by using simple random sampling. Helps to reduce sample selection bias. On the other hand, when classifying each member of a population into distinct subpopulations is not obvious, this method becomes unusable.
- **Cluster sampling** : population is divided into clusters with homogenous characteristics. Instead of randomly selecting individual members in each cluster, entire clusters are randomly chosen and each of these clusters has an equal chance of being selected as part of a sample. If clusters are large, then we can conduct a **multistage sampling** by using one of the previous sampling methods to select individual members within each cluster.

#### Non-probability sampling

- **Convenience sampling** : researchers choose members the most accessible to the researchers from a population to form a sample. This method is easy and inexpensive but generalizing the results obtained to the whole population is questionable.
- **Quota sampling** : a sample group is selected to be representative of a larger population in a non-random way. For example, recruiters with limited time can use the quota sampling method to search for potential candidates from professional social networks (LinkedIn, Indeed.com, etc.) and interview them. This method is cost-effective and saves time but presents bias during the selection process.

## Chapter 2, Distributions of Data

### Understanding data types

#### Nominal data (qualitative)

Data labeled with distinct groupings (under labels). With nominal data, there is only one operation that can be performed: **equality**. Each member of a group is equal while members from different groups are unequal.
![[Pasted image 20240204165113.png]]

#### Ordinal data (qualitative)

Nominal data that exhibits an order. University education levels are an example of ordinal data with the levels BS, MS, and PhD. The new operation that is possible here, on top of **equality**, is **ordering**, as the data can be sorted.
#### Interval data (quantitative)

Used to describe data that exists on an interval scale but does not have a clear definition of zero. 
```
Take the Celsius temperature scale, for example. The data points are numeric, and the data points are evenly spaced at an interval (for example, 20 and 40 are both 10 degrees away from 30). 22 Distributions of Data In this example of the temperature scale, the definition of 0 is arbitrary. For Celsius, 0 happens to be set at water’s freezing point, but this is an arbitrary choice made by the designers of the scale.
```
So, the interval data type supports **equality**, **ordering**, and **addition/subtraction**.
#### Ratio data (quantitative)

Like interval data, ratio data is ordered numeric data, but unlike interval data, ratio data has an absolute 0.Absolute 0 means that if the value of a ratio-type variable is zero, none of that variable exists or is present. 0 is the absolute minimum value. Ratio data also supports meaningful **multiplication/division**, making ratio data the type of data with the most supported operations.

### Measuring and describing distributions

3 categories of descriptive statistics:
- Central tendency
- Variability
- Shape

#### Measuring central tendency

- **Mode** : The most common occurring instance. A dataset can be unimodal or multimodal. In cases of multimodal continuous data, the term is looser, two peaks of a distribution can be called modes even if they don't have the same magnitude.
- **Median** : Middle value occurring when the values occur in order. Can be performed on any data aside from nominal data, as it dosent have order operation. nlike the mean, the median is not affected by outliers if outliers account for a smaller percentage of the data.
- **Mean** : the average. Valid for interval and ratio data. The mean is pulled toward values with a larger absolute value.

#### Measuring variability

- **Range** : difference between the minimum and the maximum.
- **Quartile ranges** :  Range of the quartiles of the distribution after sorting (lower, middle (50%) and higher quartile). The middle quartile range is called the **Interquartile Range (IQR)**.
- **Tukey fences** : Tukey fences are based on the IQR and defined as follows (k being arbitrary, usually 1.5): 
	- Lower fence:Q1 − k(IQR) 
	- Upper fence:Q3 + k(IQR)
- **Variance** : measure of dispersion.

#### Measuring shape

- **Skewness** : The measure of asymmetry. A distribution can be left-skewed (negative skewness value), right-skewed (positive skewness value) or non-skewed (near 0 skewness value).
	![[Pasted image 20240209125831.png]]
- **Kurtosis** : Measurement of how heavy or light the tail of distribution is relative to normal distribution. If the kurtosis value is 0 or near 0, the distribution does not exhibit kurtosis. If the kurtosis value is negative, the distribution exhibits light-tailedness, and if the kurtosis value is positive, the distribution exhibits heavy-tailedness.
	![[Pasted image 20240209125806.png]]

### Normal distribution and Central Limit Theorem

Standard normal probability distribution : data occurs close more often that further away. **= no skew, no kurtosis**

**Empirical Rule** : describes the distrib as having three pertinent standard deviations around the mean μ. It has two assumptions:
- The first, second, and third standard deviations contain 68%, 95%, and 99.7% of the measurements dispersed, respectively.
- The mean, median, and mode are all equal to each other
	![[Pasted image 20240209130545.png]]

Two common forms of a normal distribution are as follows: 
- The probability density distribution (based on random sampling)
	![[Pasted image 20240209130737.png]]
- The cumulative density distribution (based on accumulative data)
	![[Pasted image 20240209130747.png]]

#### Central Limit Theorem

postulates that if random samples of n observations are taken from a population that has a specific mean, μ, and standard deviation, σ, the sampling distribution constructed from the means of the randomly selected sub-sample distributions will approximate a normal distribution having roughly the same mean, μ, and standard deviation, calculated as ![[Pasted image 20240209131159.png]] as the population. 


### [[Bootstrapping as a demonstration of the CLT]]

### Permutations

#### Basic knowledge of permutations and combinations

**The order of objects matter in permutations while it does not for combinations.**

For exemple, someone needs to choose at random 3 people out 10 to get moneys prices (winners being Huy, Paul, and Stuart). 
In one example, he gives out 1000\$, 500\$, et 300\$. In a second, he gives three equal 500$ prices. In the first exemple, the prices are different for each winner so it plays out in more ways than the second exemple as the order prize arrangement doesn't matter, **the first example is a permutation example**.
![[Pasted image 20240209153511.png]]
In Python, the package `itertools` is used to find permutations directly with `permutations`.

For the second example, the order doesn't matter. In the first example, when the 3 winners are selected, there are six ways of  arranging the prizes, whereas there is only one way of doing so in the second example.

![[Pasted image 20240209154815.png]]

In Python, the package `itertools` is used to find permutations directly with combinations.

#### Permutation testing

While bootstrapping is useful for estimating statistical parameters, **permutations are useful for hypothesis testing**. 

Permutation testing is used to test the null hypothesis between two samples generated from the same population. It has different names such as **exact testing, randomization testing, and re-randomization testing**.

[Article constituant un meilleur exemple imo.]([Tests de permutation avec le logiciel R - DellaData](https://delladata.fr/tests-de-permutation-avec-le-logiciel-r/#:~:text=Les%20tests%20de%20permutation%20sont,reposent%20sur%20des%20distributions%20th%C3%A9oriques%20!))


### Transformations


- Log transformation 
- Square root transformation 
- Cube root transformation
![[Pasted image 20240209170853.png]]

Using transformation, we can see the transformed histograms are more normally distributed than the original one. It seems that the best transformation in this example is cube root transformation. With real-world data, it is important to determine whether a transformation is needed, and, if so, which transformation should be used.

## Chapter 3, Hypothesis Testing

## Chapter 6, [[Simple Linear Regression]]

## Chapter 7, [[Multiple Linear Regression]]
