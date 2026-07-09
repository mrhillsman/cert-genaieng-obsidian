![[14-gen-ai-advance-fine-tuning-for-llms/additional-notes/Practice-Quiz-Proximal-Policy-Optimization-PPO-Coursera-07-09-2026_00_25.png]]

The **first option** is the best answer:

> **A policy for distributing tokens conditioned on a given set of query tokens.**

In language-model RL/PPO, the **policy** is the model’s behavior as a probability distribution over possible next tokens or responses.

Think of it like this:

```text
Prompt/query: "The movie was"

Model policy:
  "great"  -> 0.45
  "bad"    -> 0.15
  "okay"   -> 0.20
  "boring" -> 0.10
  ...
```

The policy does **not** usually mean a hard-coded rule like:

```text
If prompt contains X, output Y.
```

It means:

```text
Given the current input, what probabilities does the model assign to possible outputs?
```

In PPO language:

```text
policy πθ(response | query)
```

means:

```text
the model's probability of generating a response, given the query
```

So the quiz feedback is trying to say: the policy **uses/produces samples from** the model’s output distribution. But in most RL/LLM terminology, the policy itself is commonly represented as that conditional probability distribution.

Why the selected answer was wrong:

> “A policy distribution represents all possible outputs based on the predefined rules.”

The problem is **“predefined rules.”** Language models usually do not generate from explicit rules. They generate from learned probabilities.

So choose the first option.



![[14-gen-ai-advance-fine-tuning-for-llms/additional-notes/Practice-Quiz-Proximal-Policy-Optimization-PPO-Coursera-07-09-2026_00_26.png]]

The correct answer is the **first option**:

> **Expected reward is the weighted sum of the future rewards, predicted by the reward model, given the current state and action.**

Your selected answer was too narrow because it says **immediate rewards** and averages over **all possible states**. Expected reward is more about:

```text
Given this state,
and given this action,
what reward do we expect to get in the future?
```

In RL terms, this is close to a **value estimate** or **action-value estimate**:

```text
Q(state, action) = expected future reward after taking that action in that state
```

A simple example:

```text
State: chatbot has received the prompt "Tell me something encouraging."

Action A: generate a positive response
Expected reward: high

Action B: generate a rude response
Expected reward: low
```

The “expected” part means we are not guaranteed a reward. We are estimating the likely reward based on what usually happens after that action.

Why the first option fits best:

```text
current state + action → possible future outcomes → weighted average reward
```

The weights are probabilities. Outcomes that are more likely count more in the expected reward.