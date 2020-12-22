# EAI320_Practical3
# Artificial Neural Network
## Scenario
Artificial neural networks (ANNs) refer to a framework of machine learning algorithms that
are inspired by the biological neural networks that constitute animal brains. ANNs learn to
perform tasks and make decisions by considering examples, and are generally programmed
without any task-specific rules. ANNs can approximate functions that represent continuous,
discrete or categorical data, as long as the data are appropriately encoded. This feature
means that neural nets have a wide range of applicability [3].

This assignment will require students to implement an ANN with backpropagation. The
trained ANN will be used to propose objects for a rock-paper-scissors (RPS) agent to play
during a match. The same RPS framework that was used for the previous assignment will
be used again [4].

The ANN will take as input the last 2 moves from the game, and should return a single
proposed object. This approach is similar to that of the genetic algorithm (GA) from the
previous practical [5]. The inputs and outputs of the ANN can be represented in a number
of ways, and it is up to the student to decide how they would do this. Figure 1 represents
an example of an ANN that uses a 1-of-K (also one-hot) encoding scheme [6] for the
inputs and the outputs. Remember that although the network in Figure 1 only considers
the last move, the ANN that will be implemented for this practical will consider the last
two moves.


# See practical guide for more detail description
