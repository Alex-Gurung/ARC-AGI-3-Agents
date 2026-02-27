Let's design an interesting agent for arc-agi 3 that, for now, works with a VLLM server and maybe a local instantiation of the model, on 1 gpu.

We can start with google/gemma-3-1b-it for convenience.

In my view there are three levels of abstraction for these games that are relevant: Plans, subgoals, and actions. Plans compose an approach to solving a level. Subgoals represent the underlying parts of a plan that must be accomplished, and actions represent the individual moves made. You can see how actions compose subgoals, and subgoals compose plans.

Each part of the agent will need to work on each of these levels of abstraction, building up intuition for each part.

What I'm envisioning is a three part agent:
1) curiosity, this agent proposes actions, subgoals, or plans that _teach us_ new information about the game. One way of representing this is actions, subgoals, or plans, that, when attempted, maximize the *surprise* that our agent experiences from seeing the new state.
2) learner, this agent proposes new 'lessons' to add to a persistent memory of this game, synthesizing insights from previously attempted actions, subgoals, and plans and seeing the result. For example, action 0 seems to move a box to the left (action abstraction), or touching the movable box to the highlighted square changes its color (subgoal) or making the rightmost diagram match that of the bottom left corner completes the puzzle (plan). This memory will have to have a maximum size, and support ADD, REMOVE, MODIFY operations, functioning somewhat like a database. The goal of this learner/updater or whatever we call it is to minimize the surprise of actions, subgoals, and plans, via its memory updates.
2) solver, this agent takes into account the current memory, and tries to solve the task. In a way this is the 'exploitation' part, where we have a good understanding of how the game works and how we want to try and exploit this knowledge to solve it.

I think the general loop will look something like this:

1) explore <-> update for a few steps until we settle on our understanding
2) exploit until we either complete the task or find something surprising that contradicts our assumptions/implies something new, in which case we go back to 1.
