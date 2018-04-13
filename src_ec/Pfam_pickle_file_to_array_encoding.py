#This function is for name to array encoding
def Pfam_from_pickle_file_encoding(name_list_pickle_filename,model_names_list_filename):
    import cPickle
    import numpy as np
    f=open(name_list_pickle_filename,'r')
    name_list=cPickle.load(f)
    f.close()
    f=open(model_names_list_filename,'r')
    model_list=cPickle.load(f)
    f.close()
    encoding=[]
    for i in range(len(name_list)):
        if i%1000==0:
            print('Processing %dth sequence.'%i)
        single_encoding=np.zeros(16306)
        if name_list[i] != []:
            for single_name in name_list[i]:
                single_encoding[model_list.index(single_name)]=1
        encoding.append(single_encoding)
    return encoding

if __name__ == '__main__':
    pass
