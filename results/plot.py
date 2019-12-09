import argparse
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

def arg_parse():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--data_path', type=str, default='', help='Path to the data set')
    parser.add_argument('--save_path', type=str, default='', help='Path to save graph')

    args = parser.parse_args()
    return args

def process_data (data_path):
    results = pd.read_csv(data_path)

    actor_loss = results['actor_loss'].values
    value_loss = results['value_loss'].values
    reward = results['sum_reward'].values
    num_jobs = results['num_jobs'].values
    average_job_duration = results['average_job_duration'].values

    # Mean average reward and episode reward for every 100 episodes
    reshape_val = 10
    average_actor_loss = np.mean(actor_loss.reshape(-1, reshape_val), axis=1)
    average_reward = np.mean(reward.reshape(-1, reshape_val), axis=1)
    average_job_duration = np.mean(average_job_duration.reshape(-1, reshape_val), axis=1)
    episodes = [i for i in range(0, len(actor_loss), reshape_val)]

    print(len(episodes))
    print(average_actor_loss.shape)
    
    graph_data = [(episodes, average_actor_loss), (episodes, average_reward), (episodes, average_job_duration)]
    return graph_data

def plot_prediction(graph_data, args):
    
    plt.subplot(311, xlabel='Epochs', ylabel='Actor loss')
    plt.plot(graph_data[0][0], graph_data[0][1], 'r')
    plt.legend()
    plt.subplot(312, xlabel='Epochs', ylabel='Reward')
    plt.plot(graph_data[1][0], graph_data[1][1], 'b')
    plt.legend()
    plt.subplot(313, xlabel='Epochs', ylabel='Average Job Duration (ms)')
    plt.plot(graph_data[2][0], graph_data[2][1], 'g')
    plt.legend()
    
    if args.save_path != '':
        plt.savefig(args.save_path)
    else:
        plt.show()



if __name__ == '__main__':
    args = arg_parse()
    print(args)
    
    graph_data = process_data(args.data_path)
    #print(graph_data)
    plot_prediction(graph_data, args)