import pickle
def print_history(h):
   pass 
with open('hela10_temp','rb') as f1,open('breakhis_temp','rb') as f2:
    h1=pickle.load(f1)
    h2=pickle.load(f2)
print('----hela10----')
print_history(h1)
print('--------------')
print('----breakhis----')
print_history(h2)
print('----------------')
