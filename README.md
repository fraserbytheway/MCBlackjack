# MCBlackjack
Finding optimal strategy in blackjack using monte carlo reinforcement learning

Code Considerations:
* hash map used to store states O(1) lookup
* queue used to hold the shuffled card deck, for O(1) removal
* Implements multiprocessing when running simulations for increased simulation rates
* OOP used for game logic for clean code


TODO:
* add notebook analysis of results
  * implement win rate to create 3D plot of the best hands
* implement card split functionality
* add card counting tracker
* implement bet size function to player to decide when to double down
