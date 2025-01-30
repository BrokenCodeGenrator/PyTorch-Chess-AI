PyTorch-Chess-AI

A neural network-based chess AI that plays at an estimated 1200-1500 Elo level.

This project is my first large-scale AI experiment, trained for approximately 25 hours on an RTX 2060. While the model performs decently, its accuracy could be further optimized. If you have suggestions for improvements, feel free to contribute or contact me!
Features

  ✅ UCI Compatibility – Basic implementation, but functional

  ✅ Neural Network Architecture – 128 hidden channels with 15 residual blocks

  ✅ Training Performance – Achieved an average loss of ~0.063 on Stockfish-evaluated positions (normalized between -1 and 1)

  ✅ Standalone Execution – You can run AI_engine.py directly from the terminal:

    ./AI_engine.py

  This starts the AI with basic UCI controls.

To-Do List

- Implement the stop command in UCI

- Consider rewriting the AI in C++ (still undecided)

- Improve search depth

- Increase speed

- Experiment with different model architectures and sizes
  
