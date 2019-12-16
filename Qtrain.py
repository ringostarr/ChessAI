from Gen_Game import *
from degree_freedom_king1 import *
from degree_freedom_king2 import *
from degree_freedom_queen import *
from features import *
import numpy as np
import matplotlib.pyplot as pt

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

def getreward(state1,board_size):
    x1_k1=0
    x1_k2=0
    x1_q1=0
    y1_k1=0
    y1_k2=0
    y1_q1=0
    if(np.any(state1 == 2)):
        for row in range(board_size):
            for col in range(board_size):
                if state1[row,col]==1:
                    x1_k1=row
                    y1_k1=col
                if state1[row][col]==2:
                    x1_q1=row
                    y1_q1=col
                if state1[row][col]==3:
                    x1_k2=row
                    y1_k2=col
        m_distace1=abs(x1_k2-x1_k1)+abs(y1_k2-y1_k1)
        m_distace2 =abs(x1_k2-x1_q1)+abs(y1_k2-y1_q1)

        return 1./(abs(m_distace1)+abs(m_distace2))
    else:
        return 1


def getnpmax(Q_est,state):
    if statetoindex(state) in Q_est:
        return np.max(Q_est[statetoindex(state)])
    else:
        return 0
def getargmax(Q_est,state,allowed_a,disp=False):
    #max = np.amax(Q_est[statetoindex(state)])
    if statetoindex(state) in Q_est:
        idx = np.argmax(Q_est[statetoindex(state)][0:len(allowed_a)-1])
        if(disp is True):
            print(Q_est[statetoindex(state)],idx)
        return allowed_a[idx]
    else:
        num = (state.shape[0]-1)*8
        num=num+8
        Q_est[statetoindex(state)] = np.zeros(num)
        return np.random.choice(allowed_a)
def training(size_board,g):
    Q_est ={}

    plot_rewards=[]
    s, p_k2, p_k1, p_q1 = generate_game(size_board)
    possible_queen_a = (s.shape[0] - 1) * 8
    possible_king_a = 8
    N_a = possible_king_a + possible_queen_a

    num_visits={}
    N_episodes = 1000
    map = np.array([[1, 0],
                    [-1, 0],
                    [0, 1],
                    [0, -1],
                    [1, 1],
                    [1, -1],
                    [-1, 1],
                    [-1, -1]])
    N_moves_save = np.zeros([N_episodes, 1])
    #pt.figure(figsize=(5, 5))
   # pt.xlabel("Timesteps")
    #pt.ylabel("reward")
    count=0
    for n in range(N_episodes):
        checkmate = 0  # 0 = not a checkmate, 1 = checkmate
        draw = 0  # 0 = not a draw, 1 = draw
        i = 1
        s, p_k2, p_k1, p_q1 = generate_game(size_board)
       # s=np.array([[0,0,0,0],[0,3,0,0],[0,0,0,1],[2,0,0,0]])
       # p_k2 = np.array([2,1])
       # p_k1 = np.array([2,3])
       # p_q1 = np.array([3,0])

        # Possible actions of the King
        # :return: dfK1: Degrees of Freedom of King 1, a_k1: Allowed actions for King 1, dfK1_: Squares the King1 is threatening
        dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
        # Possible actions of the Queen
        # :return: dfQ1: Degrees of Freedom of the Queen, a_q1: Allowed actions for the Queen, dfQ1_: Squares the Queen is threatening

        fQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
        # Possible actions of the enemy king
        dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)

        while checkmate == 0 and draw == 0:
            dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
            # Possible actions of the Queen
            dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
            # Possible actions of the enemy king
            dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)
            a = np.concatenate([np.array(a_q1), np.array(a_k1)],axis=None)

            # Index postions of each available action in tge list of directions in a
            allowed_a = np.where(a > 0)[0]
            predictedMove=getargmax(Q_est,s,allowed_a)
            num_visits[statetoindex(s)] = num_visits.get(statetoindex(s), 0) + 1
            explore=0
            explore = int(np.random.rand() < 1./num_visits[statetoindex(s)])  # with probability epsilon choose action at random if epsilon=0 then always choose Greedy
            if explore:
                a_agent = np.random.choice(allowed_a)
            else:
                a_agent = predictedMove
            s_new = s.copy()
            if a_agent < possible_queen_a:
                direction = int(np.ceil((a_agent + 1) / (size_board - 1))) - 1
                steps = a_agent - direction * (size_board - 1) + 1

                s_new[p_q1[0], p_q1[1]] = 0
                mov = map[direction, :] * steps
                s_new[p_q1[0] + mov[0], p_q1[1] + mov[1]] = 2
                p_q1[0] = p_q1[0] + mov[0]
                p_q1[1] = p_q1[1] + mov[1]
                N_moves_save[n-1,0] +=1

            else:
                direction = a_agent - possible_queen_a
                steps = 1

                s_new[p_k1[0], p_k1[1]] = 0
                mov = map[direction, :] * steps
                s_new[p_k1[0] + mov[0], p_k1[1] + mov[1]] = 1
                p_k1[0] = p_k1[0] + mov[0]
                p_k1[1] = p_k1[1] + mov[1]
                N_moves_save[n-1,0] +=i


                # Compute the allowed actions for the new position

                # Possible actions of the King
            dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s_new)
                # Possible actions of the Queen
            dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s_new)
                # Possible actions of the enemy king
            dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s_new, p_k1)
            alpha = 1./num_visits[statetoindex(s)]
           # a=(1-alpha)*Q_est[statetoindex(s)][np.where(allowed_a==a_agent)]
            #b= alpha*(getreward(statetoindex(s_new))+g*getnpmax(Q_est,s_new))
           # Q_est[statetoindex(s)][np.where(allowed_a==a_agent)] = a+b

            # Player 2 turn here

            # Check for draw or checkmate
            if np.sum(dfK2) == 0 and dfQ1_[p_k2[0], p_k2[1]] == 1:
                # King 2 has no freedom and it is checked
                # Checkmate and collect reward
                checkmate = 1
                reward = 1  # Reward for checkmate
                a = (1 - alpha) * Q_est[statetoindex(s)][np.where(allowed_a == a_agent)]
                b = alpha * (reward+g*getnpmax(Q_est, s_new))
                Q_est[statetoindex(s)][np.where(allowed_a == a_agent)] = a + b
                s=s_new
                break

            elif np.sum(dfK2) == 0 and dfQ1_[p_k2[0], p_k2[1]] == 0:
                # King 2 has no freedom but it is not checked
                draw = 1
                reward =0.05
                a = (1 - alpha) * Q_est[statetoindex(s)][np.where(allowed_a == a_agent)]
                b = alpha * (reward+g*getnpmax(Q_est, s_new))
                Q_est[statetoindex(s)][np.where(allowed_a == a_agent)] = a + b
                s=s_new
                break
            else:
                a = (1 - alpha) * Q_est[statetoindex(s)][np.where(allowed_a == a_agent)]
                reward = getreward(s_new,size_board)
                b = alpha *(getreward(s_new,size_board) + (g*getnpmax(Q_est, s_new)))
                Q_est[statetoindex(s)][np.where(allowed_a == a_agent)] = a + b
                # Move enemy to random safe location
                s=s_new
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
                if (i == 1000):
                    draw = 1
                    break;
                N_moves_save[n - 1, 0] += i
            #update parameters
            # Possible actions of the King
            plot_rewards.append(reward)
            s=s_new

            #pt.clf()
            #plot_it = i-i%100
            #rew = plot_rewards if plot_it < 100 else np.array(plot_rewards[:plot_it]).reshape((50, -1)).mean(axis=1)
            #pt.plot(rew,'k-')
            #pt.show()
            #pt.pause(0.001)
            i+=1
            count+=1
            dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
                # Possible actions of the Queen
            dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
            #dfQ1_ = getQueenFreedom(s,p_q1,p_k1,p_k2)
                # Possible actions of the enemy king
            dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)

    return Q_est


