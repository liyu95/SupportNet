import pickle
import os


def print_algorithm_result(algorithm_folders, entries, metrics, iterations):
    assert len(algorithm_folders) == len(entries)
    histories = []
    for algorithm_folder in algorithm_folders:
        with open(os.path.join(algorithm_folder, 'history.pkl'), 'rb') as f:
            histories.append(pickle.load(f))
    for metric in metrics:
        print(metric)
        for i, algorithm_folder in enumerate(algorithm_folders):
            print(algorithm_folder, end=',')
            for iteration in range(1, iterations[i]+1):
                if iteration < iterations[i]:
                    if metric != 'iteration_time':
                        print(histories[i][iteration]
                              [entries[i]][metric], end=',')
                    else:
                        print(histories[i][iteration][metric], end=',')
                else:
                    if metric != 'iteration_time':
                        print(histories[i][iteration]
                              [entries[i]][metric], end='\n')
                    else:
                        print(histories[i][iteration][metric], end='\n')
        print()


def print_full_data_result(full_data_folder_prefixes, full_data_folder_suffixes, full_data_iterations, entry, metrics):
    for metric in metrics:
        print(metric)
        for i, prefix in enumerate(full_data_folder_prefixes):
            print(prefix, end=',')
            for j in full_data_iterations[i]:
                algorithm_folder = prefix + \
                    '-%dclasses-%s' % (j, full_data_folder_suffixes[i])
                with open(os.path.join(algorithm_folder, 'history.pkl'), 'rb') as f:
                    history = pickle.load(f)
                    if j < full_data_iterations[i][-1]:
                        if metric != 'iteration_time':
                            print(history[1][entry][metric], end=',')
                        else:
                            print(history[1][metric], end=',')
                    else:
                        if metric != 'iteration_time':
                            print(history[1][entry][metric], end='\n')
                        else:
                            print(history[1][metric], end='\n')
        print()

# algorithms=['SupportNet','ewc','icarl','ft']
# algorithm_folders=['Aug31-mnist-%s-full2'%alg for alg in algorithms]+\
#                   ['Aug31-cifar10-%s-full2'%alg for alg in algorithms]+\
#                   ['Aug31-cifar100-%s-70epoch'%alg for alg in algorithms]
# entries=['best_plain_before_cumul','best_plain_before_cumul','best_exemplars_mean_cumul','best_plain_before_cumul']*3
# metrics=['top1_accuracy','ck_score','f1_macro']
# iterations=[5]*8+[50]*4
# full_data_folder_prefixes=['Aug31-mnist-full_data','Aug31-cifar10-full_data','Aug31-cifar100-full_data','Sep15-cifar100-full_data']
# full_data_folder_suffixes=['full2','full2','70epoch2','70epoch_contiguous']
# full_data_iterations=[[2,4,6,8,10],[2,4,6,8,10],[10,20,30,40,50,60,70,80,90,100],list(range(2,101,1))]

# print_algorithm_result(algorithm_folders,entries,metrics,iterations)
# print_full_data_result(full_data_folder_prefixes,full_data_folder_suffixes,full_data_iterations,'best_plain_before_cumul',metrics)

# print("~~~~~~~~~~~~~~~~~~~~~~timed results~~~~~~~~~~~~~~~~~~~~~~~~~~")
# algorithms=['SupportNet']
# algorithm_folders=['Sep17-cifar10-SupportNet-iteration_time','Sep17-mnist-SupportNet-iteration_time']
# entries=['best_plain_before_cumul']*2
# metrics=['iteration_time']
# iterations=[5,5]
# full_data_folder_prefixes=['Sep17-cifar10-full_data','Sep17-mnist-full_data']
# full_data_folder_suffixes=['iteration_time','iteration_time']
# full_data_iterations=[[2,4,6,8,10],[2,4,6,8,10]]
# print_algorithm_result(algorithm_folders,entries,metrics,iterations)
# print_full_data_result(full_data_folder_prefixes,full_data_folder_suffixes,full_data_iterations,'best_plain_before_cumul',metrics)
