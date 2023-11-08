# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        # Calculate the reciprocal of the distance to the nearest food
        food_distances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if food_distances:
            min_food_distance = min(food_distances)
            score += 1.0 / (min_food_distance + 1)

        # Avoid getting too close to non-scared ghosts
        for i in range(len(newGhostStates)):
            ghostState = newGhostStates[i]
            scaredTime = newScaredTimes[i]
            if scaredTime == 0 and manhattanDistance(newPos, ghostState.getPosition()) < 2:
                score -= 10.0

        # Return the final Score
        return score
       # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        # New function to check if the game should stop at given state and depth 
        def terminalTest(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        # Define the minimax function to calculate the best action and its value.
        def minimax(state, depth, agent):
            if terminalTest(state, depth):
                return self.evaluationFunction(state), None

            if agent == 0:  # Pacman turn
                best_value = -float('inf')
                best_action = None
                for action in state.getLegalActions(agent):
                    value, action_result = minimax(state.generateSuccessor(agent, action), depth, (agent + 1) % gameState.getNumAgents())
                    
                    # Now Update the best_value and best_action if we find a better value.
                    if value > best_value:
                        best_value = value
                        best_action = action
                return best_value, best_action
            
            else:  # Ghost agent turn 
                best_value = float('inf')
                 # Iterating through the legal actions available to the ghost agent.
                for action in state.getLegalActions(agent):
                    value, action_result = minimax(state.generateSuccessor(agent, action), depth + int(agent == gameState.getNumAgents() - 1), (agent + 1) % gameState.getNumAgents())
                    # Updating the best_value by taking the minimum of its current value and the value from the successor state.
                    best_value = min(best_value, value)
                return best_value, None

        # Start the minimax search from the initial game state.
        best_value, best_action = minimax(gameState, 0, 0)

        # Return the best action for Pacman to take.
        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # New function to check if the game should stop at given state and depth 
        def terminalTest(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        # Defining the alpha-beta pruning function to find the best action and its value.
        def alphaBeta(state, depth, agent, alpha, beta):
            if terminalTest(state, depth):
                return self.evaluationFunction(state), None

            best_action = None
            if agent == 0:  # Pacman's turn (maximize)
                best_value = -float('inf')
                for action in state.getLegalActions(agent):
                    # Generating the successor state after taking the current action
                    value, action_result = alphaBeta(state.generateSuccessor(agent, action), depth, (agent + 1) % gameState.getNumAgents(), alpha, beta)
                    
                    # Now Update the best_value and best_action if we find a better value.
                    if value > best_value:
                        best_value = value
                        best_action = action

                    # If best_value exceeds than beta, we prune the search and return.    
                    if best_value > beta:
                        return best_value, best_action
                    alpha = max(alpha, best_value)

            else:  # Ghost agents' turn (minimize)
                best_value = float('inf')
                for action in state.getLegalActions(agent):
                    value, action_result = alphaBeta(state.generateSuccessor(agent, action), depth + int(agent == gameState.getNumAgents() - 1), (agent + 1) % gameState.getNumAgents(), alpha, beta)
                    
                    # Updating the best_value by taking the minimum of its current value and the value from the successor state.
                    best_value = min(best_value, value)

                    # If best_value is less than alpha then we prune the search and return.
                    if best_value < alpha:
                        return best_value, None
                    beta = min(beta, best_value)
            return best_value, best_action

        # Start the alpha-beta pruning search from the initial game state.
        best_value, best_action = alphaBeta(gameState, 0, 0, -float('inf'), float('inf'))

        # Return the best action for Pacman to take.
        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # New function to check if the game should stop at given state and depth 
        def terminalTest(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        def expectimax(state, depth, agent):
            if terminalTest(state, depth):
                # If the game should stop, return the evaluation value and no action.
                return self.evaluationFunction(state), None

            if agent == 0:  # Pacman's turn (maximize)
                best_action = None
                best_value = -float('inf')
                for action in state.getLegalActions(agent):
                    # Finding the action that can maximize the value
                    value, action_result = expectimax(state.generateSuccessor(agent, action), depth, (agent + 1) % gameState.getNumAgents())
                    # Now Update the best_value and best_action if we find a better value.
                    if value > best_value:
                        best_value = value
                        best_action = action
                return best_value, best_action
            
            else:  # Ghost agents' turn (expectation)
                legal_actions = state.getLegalActions(agent)
                num_actions = len(legal_actions)
                # Calculating the expected value by taking the average of the legal actions
                expected_value = 0.0  # Initialize the expected value
                for action in legal_actions:
                # Generate a successor state after taking the current action and calculate its value
                    successor_value, _ = expectimax(state.generateSuccessor(agent, action), depth + int(agent == gameState.getNumAgents() - 1), (agent + 1) % gameState.getNumAgents())
                    expected_value += successor_value

                # Divide the accumulated value by the number of legal actions to get the average
                expected_value /= num_actions
                return expected_value, None
            
        # Now starting the expectimax search from the initial game state.
        best_value, best_action = expectimax(gameState, 0, 0)
        return best_action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    -> I initiated by gathering essential game state data, encompassing Pacman's position, food pellet locations, 
    remaining capsules, ghost positions, and their scared timers. Subsequently, I strategically assigned weight values to these factors to dictate their relative importance. 
    Positive weight was assigned to the current score to see the score maximization, while negative weights were allocated to the number of remaining food pellets and capsules to encourage their consumption. 
    Also, a positive weight was attributed to the proximity of Pacman to the nearest food pellet. I integrated a positive weight to account for the cumulative sum of scared ghost timers, motivating Pacman to target and earn points from them. 
    This approach allowed the Pacman to make informed decisions during the game, weighing score, food, capsules, and ghost threats. 
    """
    "*** YOUR CODE HERE ***"
    pacmanPosition = currentGameState.getPacmanPosition() # Pacman current positon
    foodList = currentGameState.getFood().asList() # List of food locations
    capsules = currentGameState.getCapsules() # List of remaining capsules
    ghostStates = currentGameState.getGhostStates() # information about ghosts
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates] # Time remaining for each ghost to be scared

    # Initialize weights for each factors which are in consideration
    weight_current_score = 1000
    weight_food_count = -50
    weight_capsules = -10
    weight_distance_to_food = 1
    weight_scared_bonus = 500

    # Calculating the distance to the closest food
    if foodList:
        distance_to_food = min([manhattanDistance(pacmanPosition, food) for food in foodList])
    else:
        distance_to_food = 0

    # Total number of capsules
    num_capsules = len(capsules)

    # Calculating the ghost distances
    ghost_distances = [manhattanDistance(pacmanPosition, ghost.getPosition()) for ghost in ghostStates]

    # Checking if any ghost is near Pacman
    if any(ghost_distance <= 1 for ghost_distance in ghost_distances):
        distance_to_food = 999999999  # Assigned larger number to prioritize survival

    # Calculate the bonus for eating scared ghosts
    scared_bonus = sum(scaredTimes)

    # Now Calculating the evaluation score of the weights added for each factors added.
    evaluation_score = (
        weight_current_score * currentGameState.getScore() +
        weight_food_count * len(foodList) +
        weight_capsules * num_capsules +
        weight_distance_to_food / (distance_to_food + 1) +
        weight_scared_bonus * scared_bonus
    )

    return evaluation_score

# Abbreviation
better = betterEvaluationFunction
