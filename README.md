# A Team Formation Model for Minimizing Faultlines
This repository contains the code for both measuring faultlines according to our
definition (which we present shortly) as well as the implementation of our algorithms
for creating teams with low faultline potential. 

## Measuring faultline in teams
Given a team of "k" people each with "f" different attributes, we define the 
faultline score to be the total number of bad traingles formed by the team.
A bad triangle (with respect to a feature) refers to three individuals where two
of them are similar with respect to the feature and one is dissimilar to the other
two. For instance, if we consider the occupation as a feature, then two programmers
and one graphic designer form a bad triangle. 

Bad triangles serve as a good proxy for faultines as they practically measure to 
what extent similar groups of people can oppose the others who are different from 
them. Notice that this definition allows for diversity. For instance, a programmer,
a graphic designer, and a sales person do not form a conflict trianlge. 

Note that the number of conflict traingles can be computed by considering all triangles.
However, this is time-consuming in practice. We present a more efficient way for counting
such triangles. The code for computing the faultline score for a given team can be found in
[here](https://github.com/sanazb/Faultline/blob/master/FaultlineScore.ipynb).

## Forming teams with low faultine potentails
Our second contribution is building an algorithm that partitions a group of individuals
into a set of teams that have small faultline scores. We also present two baseline algorithms.
Here is an overview of the methods presented here:

1. **Faultline-Splitter** is our proposed algorithm for solving this problem which outperforms
th eother two baselines. Faultline-Splitter works similar to the k-means algorithm. Starting
from a random partition of individuals into teams, the algorithms starts by measuring to what
extent moving a person to another group lowers the faultline score. Then, the algorithm re-assigns
the indivduals to minimize the fautlines. This process is repeated untill convergence.

2. **Greedy** algorithm works by creating a team at each step. It deos so by selecting the next
best available individual that can join a team while keeping its faultline score as low as possible.
While the first few teams created by this algorithm have a very small faultline score, the final teams
usually end up with a high-score.

3. **Clustering** simply refers our usual technique for clustering data-points. That is, the algorithm
prefers to put similar individuals in the same groups. Note that while creating a team of similar 
individuals results in a low faultline score, this method fails to recognize that diverse teams can
also have a good faultline score, and thus has far more limited options to lower the faultline score.

An implementation of all these techniques can be found [here]().
