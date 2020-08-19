import naiveSchedule
import pandas as pd
import logging,os

accuracy = [.75,.74,.76,.74]
latency = [355,100,155,60]
#accuracy = [.75,.75,.71,.74]
#latency = [200,200,200,255]
selected_models = []
def get_requirements(constraint):
        logging.info(f'accuracy[constraint],latency[constraint]')
        return float(accuracy[constraint]),latency[constraint]

def get_models(constraints):
        global inst_list, current_latency, current_cost
        print('main invoked')
        accuracy,latency = get_requirements(constraints)
        models = naiveSchedule.select_models(latency,accuracy,2,"infaas")
        cmd=f'python3.6 naiveSchedule.py -a {accuracy} -l {latency} -c 2'
        #os.system(cmd)
        print(f"************** selected models are {models}")
        return models
        #return ""

selected_models = []
df = pd.read_csv('/home/cc/ensembling/workload/short_wits_load.csv', delimiter=',')
# User list comprehension to create a list of lists from Dataframe rows
list_of_rows = [list(row) for row in df.values]
# Print list of lists i.e. rows
print(list_of_rows)
for row in list_of_rows:
        print(row)
        #num = row[1]
        num = row[1]
        constraints = row[2]
        for i in range(num):
            selected_models.append([get_models(constraints), constraints])
for i in selected_models: 
    print(len(i[0]), i)
#print(selected_models)
