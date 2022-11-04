import os
print('===========Preproccess==========\n')
my_path = os.path.expanduser('~') + 'data/'
file_list = []
for item in os.listdir(my_path + 'test_docs'):
    name = os.path.join("txtfiles", item)
    if os.path.getsize(name) == 0:
        print('%s is empty:' % name)
        os.remove(name)
    else:
        #preproccess(name) #possible bug here uncomment this line if the files are used for the first time
        file_list.append([name, os.path.getsize(name)])
file_list = sorted(file_list, key=itemgetter(1), reverse=True)