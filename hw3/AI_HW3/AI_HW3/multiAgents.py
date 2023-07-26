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
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        minGhostDistance = min([manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        scoreDiff = childGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min([manhattanDistance(pos, food) for food in currentGameState.getFood().asList()])
        newFoodsDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = 0 if not newFoodsDistances else min(newFoodsDistances)
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance

        direction = currentGameState.getPacmanState().getDirection()
        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        if scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1


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
    Your minimax agent (Part 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Begin your code (Part 1)
        # call function "selectBest"
        result = self.selectBest(gameState, 0, 0)
        return result[1]
    """
    selectBest(state):
        no-actions: return score
        pacman: return max-value(state)
        ghost:  return min_value(state)
    """
    def selectBest(self, gameState, index, depth):
        if len(gameState.getLegalActions(index))==0 or depth==self.depth:
            return gameState.getScore(), ""
        if index==0:    # pacman index = 0
            return self.max_value(gameState, index, depth)
        else:           # ghost index >= 1
            return self.min_value(gameState, index, depth)
    """
    max_value(state):
        get all legal actions and save into "actions"
        for each a in actions:
            let successor be the child state
            get the largest value of selectBest(successor)
        return max_value and its action
    """
    def max_value(self, gameState, index, depth):
        actions = gameState.getLegalActions(index)
        v = float('-inf')
        max_a = ""
        for a in actions:
            successor = gameState.getNextState(index, a)
            successor_index = index + 1
            successor_depth = depth
            if successor_index==gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1
            cur_v = self.selectBest(successor, successor_index, successor_depth)[0]
            if cur_v > v:
                v = cur_v
                max_a = a
        return v, max_a
    """
    min_value(state):
        get all legal actions and save into "actions"
        for each a in actions:
            let successor be the child state
            get the least value of selectBest(successor)
        return min_value and its action
    """
    def min_value(self, gameState, index, depth):
        actions = gameState.getLegalActions(index)
        v = float('inf')
        min_a = ""
        for a in actions:
            successor = gameState.getNextState(index, a)
            successor_index = index + 1
            successor_depth = depth
            if successor_index==gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1
            cur_v = self.selectBest(successor, successor_index, successor_depth)[0]
            if cur_v < v:
                v = cur_v
                max_a = a
        
        return v, min_a
        # End your code (Part 1)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (Part 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Begin your code (Part 2)
        # call function "selectBest"
        result = self.selectBest(gameState, 0, 0, float('-inf'), float('inf'))
        return result[1]
    """
    selectBest(state):
        no-actions: return score
        pacman: return max-value(state)
        ghost:  return min_value(state)
    """
    def selectBest(self, gameState, index, depth, alpha, beta):
        if len(gameState.getLegalActions(index))==0 or depth==self.depth:
            return gameState.getScore(), ""
        if index==0:    # pacman index = 0
            return self.max_value(gameState, index, depth, alpha, beta)
        else:           # ghost index >= 1
            return self.min_value(gameState, index, depth, alpha, beta)
    """
    max_value(state):
        get all legal actions and save into "actions"
        for each a in actions:
            let successor be the child state
            get the largest value of selectBest(successor)
            alpha = max(alpha, value)
            if value > beta: then cut directly and return Sbecause it won't use the beta side anymore.
        return max_value and its action
    """
    def max_value(self, gameState, index, depth, alpha, beta):
        actions = gameState.getLegalActions(index)
        v = float('-inf')
        max_a = ""
        for a in actions:
            successor = gameState.getNextState(index, a)
            successor_index = index + 1
            successor_depth = depth
            if successor_index==gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1
            cur_v= self.selectBest(successor, successor_index, successor_depth, alpha, beta)[0]
            if cur_v > v:
                v = cur_v
                max_a = a
            alpha = max(alpha, v)
            if v > beta:
                return v, max_a
        return v, max_a
    """
    min_value(state):
        get all legal actions and save into "actions"
        for each a in actions:
            let successor be the child state
            get the least value of selectBest(successor)
            beta = min(alpha, value)
            if value > alpha: then cut directly and return because it won't use the alpha side anymore.
        return min_value and its action
    """
    def min_value(self, gameState, index, depth, alpha, beta):
        actions = gameState.getLegalActions(index)
        v = float('inf')
        min_a = ""
        for a in actions:
            successor = gameState.getNextState(index, a)
            successor_index = index + 1
            successor_depth = depth
            if successor_index==gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1
            cur_v= self.selectBest(successor, successor_index, successor_depth, alpha, beta)[0]
            if cur_v < v:
                v = cur_v
                min_a = a
            beta = min(beta, v)
            if v < alpha:
                return v, min_a
        return v, min_a
        # End your code (Part 2)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Part 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Begin your code (Part 3)
        # call function "selectBest"
        result = self.selectBest(gameState, 0, 0)
        return result[1]
    """
    selectBest(state):
        no-actions: return score
        pacman: return max-value(state)
        ghost:  return expected_value(state)    # ghost will move stochastically
    """
    def selectBest(self, gameState, index, depth):
        if len(gameState.getLegalActions(index))==0 or depth==self.depth:
            return self.evaluationFunction(gameState), ""
        if index==0:    # pacman index = 0
            return self.max_value(gameState, index, depth)
        else:           # expectimax-ghost index >= 1
            return self.expected_value(gameState, index, depth)
    
    """
    same as Part 1's max_value
    """
    def max_value(self, gameState, index, depth):
        actions = gameState.getLegalActions(index)
        v = float('-inf')
        max_a = ""
        for a in actions:
            successor = gameState.getNextState(index, a)
            successor_index = index + 1
            successor_depth = depth
            if successor_index==gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1
            cur_v = self.selectBest(successor, successor_index, successor_depth)[0]
            if cur_v > v:
                v = cur_v
                max_a = a
        return v, max_a
    """
    expected_value(state):
        get all legal actions and save into "actions"
        calculate the probability "pr" = 1/N
        for each a in actions:
            let successor be the child state
            get the least value of selectBest(successor)
            add all the expected utilities (pr*value)
        return expected_value and its action
    """
    def expected_value(self, gameState, index, depth):
        actions = gameState.getLegalActions(index)
        v = 0
        pr = 1.0/len(actions)
        for a in actions:
            successor = gameState.getNextState(index, a)
            successor_index = index + 1
            successor_depth = depth
            if successor_index==gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1
            cur_v = self.selectBest(successor, successor_index, successor_depth)[0]
            v += pr*cur_v
        return v, ""
        # End your code (Part 3)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (Part 4).
    """
    # Begin your code (Part 4)
    """
    take these information to estimate the evalution:
    1. 1.0/closest_food: if value is low, which means closest_food is large, which means there is no food nearby,
                        should have positive but small evalution.
                         if the ghost is too close to pac-man, it will unconditionally set to a high number.
                         if the ghost is too far away from pac-man, set higher number.
    2. food_cnt: less food left, more probability win
    3. capsule_cnt:getCapsules()
    4. cur_score: the more score I got, the more probability I win.
    At last, calculate all w*f to evalution and return it. 
    """
    pacman_pos = currentGameState.getPacmanPosition()
    ghost_pos = currentGameState.getGhostPositions()
    GhostDistance = [manhattanDistance(pacman_pos, ghost) for ghost in ghost_pos]
    
    foods = currentGameState.getFood().asList()
    food_cnt = len(foods)
    FoodDistance = [manhattanDistance(pacman_pos, food) for food in foods]
    
    capsule_cnt = len(currentGameState.getCapsules())
    closest_food = 1
    cur_score = currentGameState.getScore()
    if food_cnt:
        closest_food = min(FoodDistance)
    
    for gd in GhostDistance:
        if gd > 6:
            closest_food = 100
        if gd < 2:
            closest_food = 100
    
    f=[1.0/closest_food, food_cnt, capsule_cnt, cur_score]
    w=[1, -10, -1, 10]
    evaluation = sum([fi*wi for fi, wi in zip(f, w)])
    return evaluation
    # End your code (Part 4)

# Abbreviation
better = betterEvaluationFunction
