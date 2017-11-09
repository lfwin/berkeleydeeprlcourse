#!/usr/bin/env python

import tensorflow as tf
import pickle
import numpy as np
import gym
import tflearn

#with open("expert_data.pkl", 'rb') as f:
#    expert_data = pickle.load(f)

#print(np.shape(expert_data['observations']))
#print(np.shape(expert_data['actions']))

def view_data(data, args):
    returns = []
    observations = []
    actions = []
    step_list = []
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit
    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = data['actions'][steps, :, :]
            observations.append(obs)
            actions.append(action)
            step_list.append(steps)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)


def BC(model, args):
    returns = []
    observations = []
    actions = []
    step_list = []

    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit
    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = model.predict(obs[None, :])
            observations.append(obs)
            actions.append(action)
            step_list.append(steps)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

        our_data = {'observations': np.array(observations),
                       'actions': np.array(actions), 
                       'returns': np.array(r), 
                       'steps': np.array(steps), }
        #print(expert_data)
        # pdb.set_trace()
        # f = open("expert_data/"+args.envname+".p", 'wb')
        pickle.dump(our_data, open("our_data/"+args.envname+".p", 'wb'))



def train_model(obs, action, num_input, num_output):
    import os
    X = tflearn.input_data(shape=[None, num_input])
    dense1 = tflearn.fully_connected(X, 64, activation='relu', regularizer='L2', weight_decay=0.001)
    dropout1 = tflearn.dropout(dense1, 0.8)
    dense2 = tflearn.fully_connected(dropout1, 64, activation='relu', regularizer='L2', weight_decay=0.001)
    dropout2 = tflearn.dropout(dense2, 0.8)
    net = tflearn.fully_connected(dropout2, num_output, activation='linear')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, loss='mean_square')
    model = tflearn.DNN(net, tensorboard_verbose=0)
    if os.path.exists("model/model.tfl"):
        model.load("model/model.tfl")
    model.fit(obs, action, validation_set=0.1, batch_size=32, n_epoch=40, show_metric=True, run_id="dense_model")
    model.save("model/model.tfl")
    
    return model

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int, default=5000)
    parser.add_argument('--num_rollouts', type=int, default=0,
                        help='Number of expert roll outs')
    args = parser.parse_args()
    
    #env = gym.make(args.envname)
    #max_steps = args.max_timesteps or env.spec.timestep_limit
    
    filename = "our_data/"+args.envname+".p"
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())
    
    view_data(data, args)

    #env = gym.make(args.envname)
    #num_input = env.observation_space.shape[0]
    #num_output = env.action_space.shape[0]

    #model = train_model(data['observations'], np.squeeze(data['actions']), num_input, num_output)
    #BC(model, args)
    

    




if __name__ == '__main__':
    main()
