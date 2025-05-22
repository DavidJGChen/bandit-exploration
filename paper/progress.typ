#import "@preview/charged-ieee:0.1.3": ieee
#import "@local/david:1.0.0": *

#show: ieee.with(
  title: [CS 221 Project Progress Report:\ Exploring Bandit Algorithms],
  authors: (
    (
      name: "David Chen",
      organization: [Stanford University],
      email: "dchen11@stanford.edu"
    ),
  ),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introduction

In the broad context of online decision-making under uncertainty, the class of problems known as multi-armed bandits (MABs) has provided a rich environment for insightful theoretical analysis and applicable algorithms. Multi-armed bandit problems have been studied in a wide variety of fields, ranging from computer science and statistics, to operations research and economics. Although closely related to the more general setting of reinforcement learning, the study of MABs typically has a strong focus on the classic dilemma of exploration-exploitation. Still, fundamental insights from theoretical developments have been further developed for more complex reinforcement learning settings, and simple yet effective bandit algorithms have been deployed for many practical use-cases in the industry with great success.

== Project goals and current progress.

The primary focus of this course project is to provide a comprehensive comparison of a collection of algorithms for the classic stochastic multi-armed bandit setting. Namely, I do not focus on settings such as those of contextual or adversarial bandits. I aim to accomplish this task by reproducing simulation results such as those presented in Russo and Van Roy @IDS. I also intend to include a selection of derivations on important theoretical results. In summary, comparisons will take the form of empirical analysis of regret, runtime, and qualitative observations of algorithm behavior, as well as theoretical comparison of proven regret bounds.

At the time of writing, the current progress includes a functioning framework for simulation of various bandit algorithms. Implemented algorithms include: random, $epsilon$-greedy, explore-then-commit (ETC), Bayes-UCB, and Thompson sampling. Preliminary results entail simulation on the independent beta-Bernoulli bandit setting.

== Brief overview of the multi-armed bandit problem.

I examine the stochastic multi-armed bandit problem. At each time period, an agent is allowed to choose an action (an arm) to execute, and subsequently observes a random outcome, often in the form of a scalar reward. Outcomes are associated with the specific arm, and can either be _independent or dependent_ with respect to the other arms. The true distribution of the outcomes is unknown, and thus exploration of arms is necessary in order to gather knowledge about rewards.
In this project, I restrict analysis to settings with stationary outcome distributions over time, as well as restricting the class of eligible actions to be fixed finite sets.

The objective of MAB problems is to maximize the average cumulative reward over time. Thus, a central issue that arises is that of _exploration-exploitation_, where a tradeoff is necessary in order to discover actions associated with higher rewards, while still leveraging high reward actions over the time horizon. In general, the time horizon can be infinite, but I only analyze and implement problems in a finite-time setting for this project.

A key difference between MABs and the more general reinforcement learning framework is the lack of "state". In the setting of Markov decision processes, outcomes are associated with a changing state as well as the selected action, whereas in the restricted bandit setting, any given action is assumed to produce i.i.d. outcomes when chosen in different time periods.

Typical theoretical analysis of MABs often involves the notion of _regret_, which intuitively is the expected difference in the sum of rewards between a strategy that chooses the optimal action at every round, and the actual strategy. There is also the notion of _per-period regret_, which is specific to a single round. Upper and lower bounds on regret are usually of interest for various algorithms, and much of the literature is dedicated to deriving and improving these bounds.

= Related Works
- lai and robbins 1985
- info theoretic analysis thompson sampling
- learning to optimize via IDS
- thompson 1933
- bayes ucb
- Finite-time analysis of the multiarmed bandit
problem (UCB1)
- An empirical evaluation of Thompson sampling
- bandit formulation Some aspects of the sequential design of experiments
- bandit algorithms
- slivkin bandit stuff

= Methodology

== Problem formulation

== Dataset

All "data" is generated ad-hoc during simulations. Outcomes associated with specific distributions are generated randomly using existing libraries, such as `numpy.random` and `scipy.stats`.

For example, in a Bernoulli bandit instance, the initial parameters $theta_k$ across $K$ arms are generated independently from a continuous uniform distribution on $[0,1]$ using `np.random.uniform`. The subsequent rewards are then generated using an indicator function implemented using `np.random.rand`. These are generated as needed during simulation.

== Baseline

My baseline involves implementation of simple non-adaptive algorithms for the beta-Bernoulli bandit setting. These include a random strategy, various $epsilon$-greedy strategies, and explore-then-commit.

The random strategy simply chooses an action, uniformly at random from the set of available actions, at each time-step. This is mainly chosen to demonstrate a worst-case upper bound on regret for all subsequent algorithms.

The $epsilon$-greedy algorithms involve choosing a uniformly random action at each time period with probability $epsilon_t$, and otherwise choosing the action with the highest point estimate of mean reward. Notably $epsilon_t$ can vary over time, but overall this class of algorithms is still classified into the non-adaptive exploration category, given it does not change its exploration strategy based on the realized history.

Some examples of valid choices of $epsilon_t$ are:
- Constant (ex. $epsilon_t = epsilon in [0, 1)$),
- Decaying (ex. $epsilon_t = epsilon(t) = t^(-1 slash 3)$),
- Explore-then-commit (ex. $epsilon_t = epsilon(t) = bb(1)_(t < 200)$).

These approaches are chosen as the baseline of algorithms that do not incorporate any additional notion of uncertainty into the exploration strategy, which leads to provably worse regret-bounds and demonstrably worse realized regret in simulation.

== Main approach

More interesting algorithms arise when we attempt to balance exploration-exploitation through use of confidence intervals, probability matching, or explicitly minimizing the information ratio.

A major family of algorithms in this area are the UCB algorithms, which range from frequentist algorithms such as UCB1 to the Bayesian Bayes-UCB.

On the other hand, Thompson sampling selects an action based off of the statistical possibility that it is optimal under the posterior distributions.

Finally, I examine information-directed sampling, where the action is chosen to minimize an information ratio based on the expected regret and mutual information.

These algorithms have all been shown to have much better upper-bounds for regret than the baseline algorithms, and have also demonstrated better performance in simulation.

== Evaluation

In

= Results and Discussion

#figure(
  image("fig3.png")
)

= Future Work

