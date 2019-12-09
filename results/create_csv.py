import argparse
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

def arg_parse():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--data_path', type=str, default='', help='Path to the data set')
    parser.add_argument('--save_path', type=str, default='', help='Path to save csv')

    args = parser.parse_args()
    return args

def process_data (data_path):

    with open(args.data_path, 'r') as f:
        lines = [line[:-1] for line in f]
    
    cleaned_lines = list(filter(None, lines))
    data = {'actor_loss':[], 'value_loss':[], 'sum_reward':[], 'num_jobs':[], 'average_job_duration':[]}
    #print(cleaned_lines)
    #print(len(cleaned_lines))
    s= 11
    for idx in range(0,len(cleaned_lines), s):
        line_group = cleaned_lines[idx:idx+s]
        if line_group==[]: 
            print('here')
            continue
        #print(line_group)
        data['actor_loss'].append(float(line_group[0].split(',')[1]))
        data['value_loss'].append(float(line_group[2].split(',')[1]))
        data['sum_reward'].append(float(line_group[5].split(',')[1]))
        data['num_jobs'].append(float(line_group[7].split(',')[1]))
        data['average_job_duration'].append(float(line_group[9].split(',')[1]))
        
    df = pd.DataFrame(data)
    return df


if __name__ == '__main__':
    args = arg_parse()
    print(args)
    
    csv_data = process_data(args.data_path)
    csv_data.to_csv(args.save_path)