#!/usr/bin/env python

"""
Code to run the behavioral cloning 
Example usage:
    python cloned_policy.py experts/Walker2d-v2.pkl Walker2d-v2 --render \
            --num_rollouts 20
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import pdb


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session() as sess:

        tf_util.initialize()
        
        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []

        sess = tf.Session()
        # load meta graph and restore weights of the trained DNN model
        saver = tf.train.import_meta_graph('./saved_model_ant/saved_model_ant.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./saved_model_ant/'))

        graph = tf.get_default_graph()
        # create placeholders variables and
        # create feed-dict to feed new observations   
        observation = graph.get_tensor_by_name("input/observations:0")
        # To get all the ops in the graph:
        # for op in graph.get_operations():
        #     print(op.name)
        
        #Now, access the prediction op 
        action = graph.get_tensor_by_name("output/actions_pred:0")

        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset() # dim (N,)
            obs = np.expand_dims(obs, axis=0) # dim (1,N)
            #pdb.set_trace()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                feed_dict = {observation:obs}
                action_pred = sess.run(action,feed_dict)
                obs, r, done, _ = env.step(action_pred)
                obs = np.expand_dims(obs, axis=0)# dim (1,N)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))
        
        # if not os.path.exists(expert_data_2):
        #   os.makedirs(log_dir)

        # expert_data = {'observations': np.array(observations),
        #                'actions': np.array(actions)}

        # with open(os.path.join('expert_data_2', args.envname + '.pkl'), 'wb') as f:
        #     pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
