from torch.distributions.categorical import Categorical

from memory import Memory

def select_action(p, candidate, memory: Memory):
    distribution = Categorical(p.squeeze())
    s = distribution.sample()
    if memory is not None:
        memory.logprobs.append(distribution.log_prob(s))
    return candidate[s], s


def greedy_select_action(p, candidate):
    _, index = p.squeeze().max(0)
    action = candidate[index]
    return action


# evaluate the actions
def eval_actions(p, actions):
    softmax_distibution = Categorical(p)
    ret = softmax_distibution.log_prob(actions).reshape(-1)
    entropy = softmax_distibution.entropy().mean()
    return ret, entropy