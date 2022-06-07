import os

modules_types = ['classifiers']
for module_type in modules_types:
    for file_ in os.listdir(f'model/{module_type}/'):
        filename = file_.split('.')[0]
        __import__(f'model.{module_type}.{filename}')
