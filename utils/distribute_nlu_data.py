import os
# Import the libraries
import matplotlib.pyplot as plt



list_name = []
list1= []
for filename in os.listdir('dataset/nlu'):
    
    filepath = os.path.join('dataset/nlu', filename)
    f =  open(filepath, 'r', encoding='utf-8').readlines()

    name = filename.split('.')[0]
    name = name.split('_')[1]
    list_name.append(name[:3])
    list1.append(len(f))

print(list1)
# matplotlib histogram
fig, axes = plt.subplots()
plt.bar(list_name,list1)

# Add labels
plt.title('Distribution of dataset')
plt.xlabel('Words')
plt.ylabel('Number samples')
plt.show()