"""
Questions:
- modify the score function to include depth
- add in node counts for mnx vs mnx_ab
"""

import numpy as np
import copy as cp
import math
import Qtrain
from degree_freedom_king2 import *
from degree_freedom_queen import *
from degree_freedom_king1 import *
BLANK = "_"
def statetoindex(state):
    a=""
    #print("old state:",statetoindex())
    rows = state.shape[1]
    cols = state.shape[0]
    x1_k1 = 0
    x1_k2 = 0
    x1_q1 = 0
    y1_k1 = 0
    y1_k2 = 0
    y1_q1 = 0
    for x in range(0, rows):
        for y in range(0, cols):
         k=state[x,y]
         if state[x,y]==1:
                x1_k1=x
                y1_k1=y

         if state[x,y]==2:
             x1_q1 = x
             y1_q1 = y
         if state[x,y]==3:
             x1_k2 = x
             y1_k2 = y
    resultidx = str(x1_k1)+str(y1_k1)+"-"+str(x1_q1)+str(y1_q1)+"-"+str(x1_k2)+str(y1_k2)
    return resultidx



def NodeEval(s,a_agent,size_board,p_k1,p_q1,p_k2):
    map = np.array([[1, 0],
                    [-1, 0],
                    [0, 1],
                    [0, -1],
                    [1, 1],
                    [1, -1],
                    [-1, 1],
                    [-1, -1]])

    possible_queen_a = (s.shape[0] - 1) * 8
    possible_king_a = 8
    #make THE MOVE

    s_new = s.copy()
    if a_agent < possible_queen_a:

        direction = int(np.ceil((a_agent + 1) / (size_board - 1))) - 1
        steps = a_agent - direction * (size_board - 1) + 1

        s_new[p_q1[0], p_q1[1]] = 0
        mov = map[direction, :] * steps
        s_new[p_q1[0] + mov[0], p_q1[1] + mov[1]] = 2
        p_q1[0] = p_q1[0] + mov[0]
        p_q1[1] = p_q1[1] + mov[1]


    else:
        direction = a_agent - possible_queen_a
        steps = 1

        s_new[p_k1[0], p_k1[1]] = 0
        mov = map[direction, :] * steps
        s_new[p_k1[0] + mov[0], p_k1[1] + mov[1]] = 1
        p_k1[0] = p_k1[0] + mov[0]
        p_k1[1] = p_k1[1] + mov[1]
    #evaluate and return new state , our evaluation is just to check which state offers best reward

    return s_new,Qtrain.getreward(s,size_board)

def DepthEval(s,allowed_a,np_k1,np_q1,np_k2,size_board):
    bestval =-1
    bestaction=0
    possible_queen_a = (s.shape[0] - 1) * 8
    possible_king_a = 8
    map = np.array([[1, 0],
                    [-1, 0],
                    [0, 1],
                    [0, -1],
                    [1, 1],
                    [1, -1],
                    [-1, 1],
                    [-1, -1]])
    for i in range(len(allowed_a)-1):

        a_agent = allowed_a[i]
        s_new = s.copy()
        p_q1 =np.asarray(np.where(s==2))
        p_k1 = np.asarray(np.where(s==1))
        p_k2 = np.asarray(np.where(s==3))
        if a_agent < possible_queen_a:

            direction = int(np.ceil((a_agent + 1) / (size_board - 1))) - 1
            steps = a_agent - direction * (size_board - 1) + 1

            s_new[p_q1[0], p_q1[1]] = 0
            mov = map[direction, :] * steps
            s_new[p_q1[0] + mov[0], p_q1[1] + mov[1]] = 2
            p_q1[0] = p_q1[0] + mov[0]
            p_q1[1] = p_q1[1] + mov[1]


        else:
            direction = a_agent - possible_queen_a
            steps = 1

            s_new[p_k1[0], p_k1[1]] = 0
            mov = map[direction, :] * steps
            s_new[p_k1[0] + mov[0], p_k1[1] + mov[1]] = 1
            p_k1[0] = p_k1[0] + mov[0]
            p_k1[1] = p_k1[1] + mov[1]
        if(Qtrain.getreward(s_new,size_board)>bestval):
            bestaction=a_agent

    return bestaction



def mnx_ML(size_board,Q_est):
    """
    Minimax search for Chess , includes Q learning as maximizing player
    """
    map = np.array([[1, 0],
                    [-1, 0],
                    [0, 1],
                    [0, -1],
                    [1, 1],
                    [1, -1],
                    [-1, 1],
                    [-1, -1]])
    s, p_k2, p_k1, p_q1 = Qtrain.generate_game(size_board)
    possible_queen_a = (s.shape[0] - 1) * 8
    possible_king_a = 8
    dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
    # Possible actions of the Queen
    # :return: dfQ1: Degrees of Freedom of the Queen, a_q1: Allowed actions for the Queen, dfQ1_: Squares the Queen is threatening

    fQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
    # Possible actions of the enemy king
    dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)


    checkmate=0
    n=0
    while checkmate==0 or n==1000:
        #player 1 always starts
        n+=1
        a = np.concatenate([np.array(a_q1), np.array(a_k1)])

        # Index postions of each available action in tge list of directions in a
        allowed_a = np.where(a > 0)[0]
        a_agent = Qtrain.getargmax(Q_est,s,allowed_a,False)
        if a_agent < possible_queen_a:
            direction = int(np.ceil((a_agent + 1) / (size_board - 1))) - 1
            steps = a_agent - direction * (size_board - 1) + 1

            s[p_q1[0], p_q1[1]] = 0
            mov = map[direction, :] * steps
            s[p_q1[0] + mov[0], p_q1[1] + mov[1]] = 2
            p_q1[0] = p_q1[0] + mov[0]
            p_q1[1] = p_q1[1] + mov[1]


        else:
            direction = a_agent - possible_queen_a
            steps = 1

            s[p_k1[0], p_k1[1]] = 0
            mov = map[direction, :] * steps
            s[p_k1[0] + mov[0], p_k1[1] + mov[1]] = 1
            p_k1[0] = p_k1[0] + mov[0]
            p_k1[1] = p_k1[1] + mov[1]

        #player change , player 2 turn
        dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
        # Possible actions of the Queen
        dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
        # Possible actions of the enemy king
        dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)
        #check win/checmate
        if np.sum(dfK2) == 0 and dfQ1_[p_k2[0], p_k2[1]] == 1:
            # King 2 has no freedom and it is checked
            # Checkmate and collect reward
            checkmate = 1
            return checkmate
            break
            #check draw/lose (not checkmates)
        elif np.sum(dfK2) == 0 and dfQ1_[p_k2[0], p_k2[1]] == 0:
            # King 2 has no freedom but it is not checked
            checkmate=-1
            return checkmate
            break
        else:
            #game continues , make random enemy move
            # Move enemy King randomly to a safe location
            allowed_enemy_a = np.where(a_k2 > 0)[0]
            a_help = int(np.ceil(np.random.rand() * allowed_enemy_a.shape[0]) - 1)
            a_enemy = allowed_enemy_a[a_help]

            direction = a_enemy
            steps = 1

            s[p_k2[0], p_k2[1]] = 0
            mov = map[direction, :] * steps
            s[p_k2[0] + mov[0], p_k2[1] + mov[1]] = 3

            p_k2[0] = p_k2[0] + mov[0]
            p_k2[1] = p_k2[1] + mov[1]

            # Possible actions of the King
        dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
            # Possible actions of the Queen
        dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
            # Possible actions of the enemy king
        dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)
        if n==1000:
            return -1





def mnx_no_ML(size_board):
    """
    Minimax search for Chess

    """
    map = np.array([[1, 0],
                    [-1, 0],
                    [0, 1],
                    [0, -1],
                    [1, 1],
                    [1, -1],
                    [-1, 1],
                    [-1, -1]])
    s, p_k2, p_k1, p_q1 = Qtrain.generate_game(size_board)
    possible_queen_a = (s.shape[0] - 1) * 8
    possible_king_a = 8
    dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
    # Possible actions of the Queen
    # :return: dfQ1: Degrees of Freedom of the Queen, a_q1: Allowed actions for the Queen, dfQ1_: Squares the Queen is threatening

    fQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
    # Possible actions of the enemy king
    dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)

    checkmate = 0
    n = 0
    n = 0
    while checkmate == 0 or n == 1000:
        # player 1 always starts
        n += 1
        a = np.concatenate([np.array(a_q1), np.array(a_k1)])

        # Index postions of each available action in tge list of directions in a
        allowed_a = np.where(a > 0)[0]
        state=s.copy()
        a_agent = DepthEval(state,allowed_a,p_k1,p_q1,p_k2,size_board)

        if a_agent < possible_queen_a:
            direction = int(np.ceil((a_agent + 1) / (size_board - 1))) - 1
            steps = a_agent - direction * (size_board - 1) + 1

            s[p_q1[0], p_q1[1]] = 0
            mov = map[direction, :] * steps
            s[p_q1[0] + mov[0], p_q1[1] + mov[1]] = 2
            p_q1[0] = p_q1[0] + mov[0]
            p_q1[1] = p_q1[1] + mov[1]


        else:
            direction = a_agent - possible_queen_a
            steps = 1

            s[p_k1[0], p_k1[1]] = 0
            mov = map[direction, :] * steps
            s[p_k1[0] + mov[0], p_k1[1] + mov[1]] = 1
            p_k1[0] = p_k1[0] + mov[0]
            p_k1[1] = p_k1[1] + mov[1]

        # player change , player 2 turn
        dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
        # Possible actions of the Queen
        dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
        # Possible actions of the enemy king
        dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)
        # check win/checmate
        if np.sum(dfK2) == 0 and dfQ1_[p_k2[0], p_k2[1]] == 1:
            # King 2 has no freedom and it is checked
            # Checkmate and collect reward
            checkmate = 1
            return checkmate
            break
            # check draw/lose (not checkmates)
        elif np.sum(dfK2) == 0 and dfQ1_[p_k2[0], p_k2[1]] == 0:
            # King 2 has no freedom but it is not checked
            checkmate = -1
            return checkmate
            break
        else:
            # game continues , make random enemy move
            # Move enemy King randomly to a safe location
            allowed_enemy_a = np.where(a_k2 > 0)[0]
            a_help = int(np.ceil(np.random.rand() * allowed_enemy_a.shape[0]) - 1)
            a_enemy = allowed_enemy_a[a_help]

            direction = a_enemy
            steps = 1

            s[p_k2[0], p_k2[1]] = 0
            mov = map[direction, :] * steps
            s[p_k2[0] + mov[0], p_k2[1] + mov[1]] = 3

            p_k2[0] = p_k2[0] + mov[0]
            p_k2[1] = p_k2[1] + mov[1]

            # Possible actions of the King
        dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
        # Possible actions of the Queen
        dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
        # Possible actions of the enemy king
        dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)
        if n == 1000:
            return -1


def test1():
    # 50 tests with board size 3x3 and gamma=0.05
    Qest = Qtrain.training(4,0.05)
    wins=0
    for i in range(0,100):
        result=mnx_ML(4,Qest)
        if(result==1):
            wins+=1

    print("\nPlayed 100 Games with Q-Learning AI\n ")
    print("Games won : ",wins)
    percentage = wins*100/100
    print("\n WinRate :",percentage)
    wins=0
    for i in range(0,100):
        result=mnx_no_ML(4)
        if(result==1):
            wins+=1

    print("\nPlayed 100 Games without Q-Learning AI\n ")
    print("Games won : ",wins)
    percentage = wins*100/100
    print("\n WinRate :",percentage)
def test2():
    # 50 tests with board size 3x3 and gamma=0.05
    Qest = Qtrain.training(5,0.05)
    wins=0
    for i in range(0,100):
        result=mnx_ML(5,Qest)
        if(result==1):
            wins+=1

    print("\nPlayed 100 Games with Q-Learning AI\n ")
    print("Games won : ",wins)
    percentage = wins*100/100
    print("\n WinRate :",percentage)
    wins=0
    for i in range(0,100):
        result=mnx_no_ML(5)
        if(result==1):
            wins+=1

    print("\nPlayed 100 Games without Q-Learning AI\n ")
    print("Games won : ",wins)
    percentage = wins*100/100
    print("\n WinRate :",percentage)
def test3():
    # 50 tests with board size 3x3 and gamma=0.05
    Qest = Qtrain.training(5,0.01)
    wins=0
    for i in range(0,100):
        result=mnx_ML(5,Qest)
        if(result==1):
            wins+=1

    print("\nPlayed 100 Games with Q-Learning AI\n ")
    print("Games won : ",wins)
    percentage = wins*100/100
    print("\n WinRate :",percentage)
    wins=0
    for i in range(0,100):
        result=mnx_no_ML(5)
        if(result==1):
            wins+=1

    print("\nPlayed 100 Games without Q-Learning AI\n ")
    print("Games won : ",wins)
    percentage = wins*100/100
    print("\n WinRate :",percentage)
def test4():
    # 50 tests with board size 3x3 and gamma=0.05
    Qest = Qtrain.training(4,0.01)
    wins=0
    for i in range(0,100):
        result=mnx_ML(4,Qest)
        if(result==1):
            wins+=1

    print("\nPlayed 100 Games with Q-Learning AI\n ")
    print("Games won : ",wins)
    percentage = wins*100/100
    print("\n WinRate :",percentage)
    wins=0
    for i in range(0,100):
        result=mnx_no_ML(4)
        if(result==1):
            wins+=1

    print("\nPlayed 100 Games without Q-Learning AI\n ")
    print("Games won : ",wins)
    percentage = wins*100/100
    print("\n WinRate :",percentage)
def test5():
    # 50 tests with board size 3x3 and gamma=0.05
    Qest = Qtrain.training(6,0.05)
    wins=0
    for i in range(0,100):
        result=mnx_ML(6,Qest)
        if(result==1):
            wins+=1

    print("\nPlayed 100 Games with Q-Learning AI\n ")
    print("Games won : ",wins)
    percentage = wins*100/100
    print("\n WinRate :",percentage)
    wins=0
    for i in range(0,100):
        result=mnx_no_ML(6)
        if(result==1):
            wins+=1

    print("\nPlayed 100 Games without Q-Learning AI\n ")
    print("Games won : ",wins)
    percentage = wins*100/100
    print("\n WinRate :",percentage)
if __name__ == "__main__":
    #test1()
    #test2()
    test3()
    #test4()
    #test5()



