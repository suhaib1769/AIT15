class Bayes:
    def __init__(self, hypotheses, priors, observations, likelihoods):
        self.hypotheses = hypotheses
        self.priors = priors
        self.observations = observations
        self.likelihoods = likelihoods

    def print_init_values(self):
        print(f"List of Hypotheses: {self.hypotheses}")
        print(f"List of Prior probabilities: {self.priors}")
        print(f"List of observations: {self.observations}")
        print(f"List of likelihoods: {self.likelihoods}")

    def likelihood(self, observation, hypothesis):
        hypo_index = self.hypotheses.index(hypothesis)
        obv_index = self.observations.index(observation)
        return self.likelihoods[hypo_index][obv_index]

    def norm_constant(self, observation):
        norm_c = 0
        obv_index = self.observations.index(observation)
        for i in range(len(self.hypotheses)):
            norm_c += self.priors[i] * self.likelihoods[i][obv_index]
        return norm_c

    def single_posterior_update(self, observation, priors):
        posteriors = []
        norm_c = self.norm_constant(observation)

        for i in range(len(priors)):
            likelihood = self.likelihood(observation, self.hypotheses[i])
            posteriors.append(priors[i]*likelihood/norm_c)

        return posteriors

    def compute_posterior(self, obv_list): # have to check formula (right now: recursively update the posterior probabilities so that previous posterior becomes current prior)
        posterior = self.single_posterior_update(obv_list[0], self.priors)
        for i in range(1, len(obv_list)):
            posterior = self.single_posterior_update(obv_list[i], posterior)
        return posterior
    
    