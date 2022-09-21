# Part 1: data
from bayes import *

cookies = Bayes(
    hypotheses = ["Bowl1", "Bowl2"], 
    priors = [0.5, 0.5], 
    observations = ["chocolate", "vanilla"], 
    likelihoods = [
        [15/50, 35/50], 
        [30/50, 20/50]
    ]
)

# Part 2: data
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
probs = archery.compute_posterior(["Yellow", "White", "Red", "Red", "Blue"])
answers.append(probs[archery.hypotheses.index('Intermediate')])
most_likely_level = archery.hypotheses[probs.index(max(probs))]
answers.append(most_likely_level)

def file_creator():
    """Function that creates and writes the answers to the questions in a .txt file named: group15.txt
    """
    with open("group_15.txt", 'w') as file:
        answer_string = ""
        for i in range(len(answers)):
            answer_string += "Answer question {}: {}\n".format(i+1, answers[i] if isinstance(answers[i], str) else round(answers[i], 3))
        file.write(answer_string)

if __name__ == "__main__":
    file_creator()