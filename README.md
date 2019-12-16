# ChessAI
Gen_game.py - Responsible for generating game instances
degree_freedom_king1.py - helper for Gen_game.py,provides actions for King
degree_freedom_queen.py - helper for Gen_game.py,provides actions for queen
degree_freedom_king2.py - helper for Gen_game.py,provides actions for enemy king

All 4 files are from the following link:
https://github.com/PhilipAD/Chess-Q-learning-Reinforcement-Learning


--QTrain.py--
file responsible for training and generating Qvalues for each state

Functions: 

statetoIndex(state): takes np.array as input and return unique id for that state as a string.

getreward(state,boardsize): returns the reward offered at a particular state

gerargmax(Qest,state,allowed_actions,disp=False): returns best action after looking at TD estimate for each action.

getnpmax(Qest,state): returns np.max(Q(s,a))

training(boardsize,gamma): responsible for main training , takes boardsize and gamma as input and builds Qvalues for each state visited. Returns Qvalues at the end.


--TreeSearch.py--
Main file responsible for running test and incorporating Q-Learning with treeSearch.

Functions-
NodeEval(state,action,size,positions of pieces) : one step look ahead to determine which action is best. Used for standard non AI game playing , returns reward and new state for an action

DepthEval(state,actions): Returns best action after looking ahead one step , uses NodeEval to get rewards for that state.

mnx_ML(size,Qest):Main MinMaxtreeSeach with Q-Learning AI component ,uses Q values to assign values to each state and maximizes this for player

mnx_no_ML(size): MinMaxtreeSeach without Q-Learning AI component ,uses NodeEval values to assign values to each state and maximizes this for player

TESTS:
5 tests , testing different board size and gamma values

Main():
calls test functions

