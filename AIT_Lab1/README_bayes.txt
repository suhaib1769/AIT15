This folder contains the solutions of group 15 to the Bayes assignment of CS4375.
The class Bayes calculates posterior probabilities, when the priors, likelihood function and observations are provided.

When initializing an instance of the Bayes class, we provide the arguments 'hypotheses', 'priors', 'observations' and 'likelihoods'.
These are stored in the appropriate attributes.

The method "likelihood" takes an observation O and hypothesis  H and returns the likelihood P(O|H) that it retrieves from its stored 'likelihoods'.

The method "norm_constant" takes an observation and computes P(O) by summing over the hypotheses the product of the prior of H P(H) and the likelihoods P(O|H).

The method "single_posterior_update" takes an observation and a list of priors.
It computes for each of the priors the posterior probability P(O|H) by calling "likelihood" for each.
These posteriors are stored in a list 'posteriors', which is returned.

Finally, "compute_posterior" takes a list of observations and returns the posterior probabilities based on these observations.]
It does so by calling "single_posterior_update" repeatedly, each time taking the next observation from the list.
In the first iteration, it provides the stored 'priors' as priors.
Since we keep updating the posteriors, each next iteration takes the current value of the posterior as its prior.
After all observations have been processed, the posterior probabilities are returned.

To run our code from the terminal and generate the answer file called "group_15.txt", go to the folder where the python files are saved.
The python file that produces the answer file is then run by typinh "python .\AIT_lab1.py".