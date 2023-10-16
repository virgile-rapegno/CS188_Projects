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
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
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
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newGhostPosition = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"

        """INF = float("inf")

        closestGhost = min(
            [
                manhattanDistance(newPos, ghostState.getPosition())
                for ghostState in newGhostStates
            ]
        )

        def foodIsEaten():
            if currentGameState.getFood().count() > newFood.count():
                return True
            return False

        def ghostIsScared():
            if min(newScaredTimes) > 0:
                return True
            return False

        def ghostIsScaredNow():
            if min(newScaredTimes) > 0 and min(currentScaredTimes) == 0:
                return True
            return False

        if closestGhost < 2:
            return -INF

        res = 0
        res -= min([manhattanDistance(newPos, food) for food in newFood])
        if not ghostIsScared():
            res += closestGhost
        if foodIsEaten():
            res += 100
        if ghostIsScaredNow():
            res += 100
        return res"""

        if len(newFood.asList()) == 0:
            return 1000
        minDistanceFood = min(
            [manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
        )

        minDistanceGhost = min(
            [manhattanDistance(newPos, ghostPos) for ghostPos in newGhostPosition]
        )

        scaredTime = min(newScaredTimes)

        def foodIsEaten():
            if currentGameState.getFood().count() > newFood.count():
                return True
            return False

        res = 0
        if action == Directions.STOP:
            res -= 50
        res -= minDistanceFood * 10
        if scaredTime == 0:
            res += minDistanceGhost
            if minDistanceGhost < 4:
                res += minDistanceGhost * 50
        else:
            res += 100
        if foodIsEaten():
            res += 100

        return res


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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
        inf = float("inf")

        def value(agentsList, gameState):
            if gameState.isWin() or gameState.isLose() or len(agentsList) == 0:
                return self.evaluationFunction(gameState)
            if agentsList[0] == 0:
                return maxValue(agentsList, gameState)
            return minValue(agentsList, gameState)

        def maxValue(agentsList, gameState):
            v = -inf
            currentAgent = agentsList[0]
            for action in gameState.getLegalActions(currentAgent):
                nextState = gameState.generateSuccessor(currentAgent, action)
                nextValue = value(agentsList[1:], nextState)
                v = max(v, nextValue)
            return v

        def minValue(agentsList, gameState):
            v = +inf
            currentAgent = agentsList[0]
            for action in gameState.getLegalActions(currentAgent):
                nextState = gameState.generateSuccessor(currentAgent, action)
                nextValue = value(agentsList[1:], nextState)
                v = min(v, nextValue)
            return v

        def agentsListGenerator():
            return [
                i for _ in range(self.depth) for i in range(gameState.getNumAgents())
            ]

        agentsList = agentsListGenerator()
        bestAction = None
        bestV = -inf
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextValue = value(agentsList[1:], nextState)
            if nextValue > bestV:
                bestV = nextValue
                bestAction = action
        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        inf = float("inf")
        alpha = -inf
        beta = +inf

        def value(agentsList, gameState, alpha, beta):
            if gameState.isWin() or gameState.isLose() or len(agentsList) == 0:
                return self.evaluationFunction(gameState)
            if agentsList[0] == 0:
                return maxValue(agentsList, gameState, alpha, beta)
            return minValue(agentsList, gameState, alpha, beta)

        def maxValue(agentsList, gameState, alpha, beta):
            v = -inf
            currentAgent = agentsList[0]
            for action in gameState.getLegalActions(currentAgent):
                nextState = gameState.generateSuccessor(currentAgent, action)
                nextValue = value(agentsList[1:], nextState, alpha, beta)
                v = max(v, nextValue)
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def minValue(agentsList, gameState, alpha, beta):
            v = +inf
            currentAgent = agentsList[0]
            for action in gameState.getLegalActions(currentAgent):
                nextState = gameState.generateSuccessor(currentAgent, action)
                nextValue = value(agentsList[1:], nextState, alpha, beta)
                v = min(v, nextValue)
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        def agentsListGenerator():
            return [
                i for _ in range(self.depth) for i in range(gameState.getNumAgents())
            ]

        agentsList = agentsListGenerator()
        bestAction = None
        bestV = -inf
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextValue = value(agentsList[1:], nextState, alpha, beta)
            if nextValue > bestV:
                bestV = nextValue
                bestAction = action
            if bestV > beta:
                return bestAction
            alpha = max(alpha, bestV)
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        inf = float("inf")

        def value(agentsList, gameState):
            if gameState.isWin() or gameState.isLose() or len(agentsList) == 0:
                return self.evaluationFunction(gameState)
            if agentsList[0] == 0:
                return maxValue(agentsList, gameState)
            return expValue(agentsList, gameState)

        def maxValue(agentsList, gameState):
            v = -inf
            currentAgent = agentsList[0]
            for action in gameState.getLegalActions(currentAgent):
                nextState = gameState.generateSuccessor(currentAgent, action)
                nextValue = value(agentsList[1:], nextState)
                v = max(v, nextValue)
            return v

        def expValue(agentsList, gameState):
            v = 0
            currentAgent = agentsList[0]
            nextActions = gameState.getLegalActions(currentAgent)
            probability = 1 / len(nextActions)
            for action in nextActions:
                nextState = gameState.generateSuccessor(currentAgent, action)
                nextValue = value(agentsList[1:], nextState)
                v += probability * nextValue
            return v

        def agentsListGenerator():
            return [
                i for _ in range(self.depth) for i in range(gameState.getNumAgents())
            ]

        agentsList = agentsListGenerator()
        bestAction = None
        bestV = -inf
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextValue = value(agentsList[1:], nextState)
            if nextValue > bestV:
                bestV = nextValue
                bestAction = action
        return bestAction


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    Pos = currentGameState.getPacmanPosition()
    Score = currentGameState.getScore()
    Food = currentGameState.getFood().asList()
    GhostStates = currentGameState.getGhostStates()
    GhostPosition = currentGameState.getGhostPositions()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    x, y = Pos

    FoodDistances = [manhattanDistance(Pos, foodPos) for foodPos in Food]
    minDistanceFood = 0
    if len(FoodDistances) > 0:
        minDistanceFood = min(FoodDistances)

    minDistanceGhost = min(
        [manhattanDistance(Pos, ghostPos) for ghostPos in GhostPosition]
    )

    minScaredTimes = min(ScaredTimes + [1])
    if minScaredTimes > 0:
        minDistanceGhost += 10

    # Comme on ne peut pas utiliser MazeDistance, Pacman se coince en utilisant
    # la distance de manhattan donc on lui fait éviter les murs
    wall = 0
    if currentGameState.hasWall(x - 1, y):
        wall += 1
    if currentGameState.hasWall(x + 1, y):
        wall += 1
    if currentGameState.hasWall(x, y - 1):
        wall += 1
    if currentGameState.hasWall(x, y + 1):
        wall += 1

    return (
        (10 * Score)
        - (10 * minDistanceFood)
        + (11 * minDistanceGhost)
        - (5 * wall)
        - (5 * len(Food))
    )


# Abbreviation
better = betterEvaluationFunction
