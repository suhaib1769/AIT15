# Part 1: Bayes class
class Bayes:

    """
    A class used to represent a probability problem to be solved using Bayes theorem

    ...

    Attributes
    ----------
    hypotheses : list(str)
        a list of all of the possible hypotheses for the problem
    priors : list(float)
        a list of prior probabilities for all hypotheses H, (P(H))
    obersvations : list(str)
        a list of all of the possible observations for the problem
    likelihoods : list(float)
        a list of likelihood probabilities for all observations O given a hypothesis H, P(O|H)
    posteriors: list(float)
        a list of all posterior probabilities for all hypotheses given a (set of) observation(s) (P(H|O))

    Methods
    -------
    print_init_values()
        Prints all initial values for the attributes

    likelihood(observation, hypothesis)
        Returns the likelihood probability (P(O|H)) for a given pair of observation O and hypothesis H
    
    norm_constant(observation)
        Returns the normalizing constant for a given observation

    single_posterior_update(observation, priors)
        Updates and returns the posterior probabilities of hypotheses for a given observation

    compute_posterior(obv_list)
        Returns the posterior probabilities of all hypotheses for a given sequence (list) of observations

    """

    def __init__(self, hypotheses, priors, observations, likelihoods):

        """
        Parameters
        ----------
        hypotheses : list(str)
            a list of all of the possible hypotheses for the problem
        priors : list(float)
            a list of prior probabilities for all hypotheses H, (P(H))
        obersvations : list(str)
            a list of all of the possible observations for the problem
        likelihoods : list(float)
            a list of likelihood probabilities for all observations O given a hypothesis H, P(O|H)
        """

        self.hypotheses = hypotheses
        self.priors = priors
        self.observations = observations
        self.likelihoods = likelihoods
        self.posteriors = None

    def print_init_values(self):

        """Prints all initial values for the attributes.

        Parameters
        ----------
        None
        """        

        print(f"List of Hypotheses: {self.hypotheses}")
        print(f"List of Prior probabilities: {self.priors}")
        print(f"List of observations: {self.observations}")
        print(f"List of likelihoods: {self.likelihoods}")
        print(f"List of posterior probabilities: {self.posteriors}")

    def likelihood(self, observation, hypothesis):

        """Returns the likelihood probability (P(O|H)) for a given pair of observation O and hypothesis H
        Parameters
        ----------
        observation: str
            The observation that has been made
        hypothesis: str
            The hypothesis that is being assumed to be true
        """ 

        hypo_index = self.hypotheses.index(hypothesis)
        obv_index = self.observations.index(observation)
        return self.likelihoods[hypo_index][obv_index]

    def norm_constant(self, observation):

        """Returns the normalizing constant for a given observation
        Parameters
        ----------
        observation: str
            The observation that has been made
        """ 

        norm_c = 0
        obv_index = self.observations.index(observation)
        for i in range(len(self.hypotheses)):
            norm_c += self.priors[i] * self.likelihoods[i][obv_index]
        return norm_c

    def single_posterior_update(self, observation, priors):

        """Updates and returns the posterior probabilities of hypotheses for a given observation
        ----------
        observation: str
            The observation that has been made
        priors: list(float)
            a list of prior probabilities for all hypotheses H, (P(H))
        """ 
        self.posteriors = []

        norm_c = self.norm_constant(observation)

        for i in range(len(priors)):
            likelihood = self.likelihood(observation, self.hypotheses[i])
            self.posteriors.append(priors[i]*likelihood/norm_c)

        return self.posteriors

    def compute_posterior(self, obv_list): 

        """Returns the posterior probabilities of all hypotheses for a given sequence (list) of observations
        ----------
        obv_list: list(str)
            List of observations that were made in a sequence
        """ 

        init_priors = self.priors
        for i in range(0, len(obv_list)):
            self.priors = self.single_posterior_update(obv_list[i], self.priors)
        posteriors = self.priors
        self.priors = init_priors
        return posteriors

# Initializing Bayes class instances 
cookies = Bayes(
    hypotheses = ["Bowl1", "Bowl2"], 
    priors = [0.5, 0.5], 
    observations = ["chocolate", "vanilla"], 
    likelihoods = [
        [15/50, 35/50], 
        [30/50, 20/50]
    ]
)

archery = Bayes(
    hypotheses=["Beginner", "Intermediate", "Advanced", "Expert"],
    priors = [0.25, 0.25, 0.25, 0.25],
    observations=["Yellow", "Red", "Blue", "Black", "White"],
    likelihoods=[
        [0.05, 0.1, 0.4, 0.25, 0.2],
        [0.1, 0.2, 0.4, 0.2, 0.1],
        [0.2, 0.4, 0.25, 0.1, 0.05],
        [0.3, 0.5, 0.125, 0.05, 0.025]
    ]
)

answers = [] #list for answer values

# Computation of part 1: cookies
l = cookies.likelihood("chocolate", "Bowl1")
n_c = cookies.norm_constant("vanilla")
p_1 = cookies.single_posterior_update("vanilla", [0.5, 0.5])
answers.append(p_1[0])
p_2 = cookies.compute_posterior(["chocolate", "vanilla"])
p_2_answer = cookies.compute_posterior(["chocolate","vanilla"])[cookies.hypotheses.index('Bowl2')]
answers.append(p_2_answer)

# Computation of part 2: archery
probs = archery.compute_posterior(["Yellow", "White", "Blue", "Red", "Red", "Blue"])
answers.append(probs[archery.hypotheses.index('Intermediate')])
most_likely_level = archery.hypotheses[probs.index(max(probs))]
answers.append(most_likely_level)

def file_creator():
    """Function that creates and writes the answers to the questions in a .txt file named: group15.txt
    """
    with open("group_15.txt", 'w') as file:
        answer_string = ""
        for i in range(len(answers)):
            answer_string += "{}\n".format(answers[i] if isinstance(answers[i], str) else round(answers[i], 3))
        file.write(answer_string)

if __name__ == "__main__":
    file_creator()