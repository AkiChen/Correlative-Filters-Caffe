#!/usr/bin/python
import sys
import re
def append(file_name):                                                                                                                                                         
                                                                                                                                                                               
    updated_file = 'empty'
    with open(file_name, 'r') as f:
        full_text = f.read()                                                                                                                                                   
        accuracy_layer = re.search(r'layers?\s\{[^}]*type:?\s"Accuracy"[^}]*\}',full_text)
        label_layer_name = re.search(r'bottom:?\s\"((?!label)\w+)\"', accuracy_layer.group(0))                                                                                 
                                                                                                                                                                               
        added_layer = 'layer {\nname: "softmax"\ntype: "Softmax"\nbottom: "bottom_name"\ntop: "softmax"\ninclude { phase: TEST }\n}'
        added_layer = added_layer.replace('bottom_name',label_layer_name.group(1))

        false_multi_view_line = re.search(r'multi_view:\s*false',full_text)
        true_multi_view_line = false_multi_view_line.group(0).replace('false','true')                                                                                          
                                                                                                                                                                               
        full_text = full_text + added_layer                                                                                                                                    
        full_text = full_text.replace(false_multi_view_line.group(0), true_multi_view_line)

        updated_file = full_text

    output = open(file_name, 'w')
    output.write(updated_file)                                                                                                                                                 
    output.close()

def remove(file_name):                                                                                                                                                         

    updated_file = 'empty'
    with open(file_name, 'r') as f:
        full_text = f.read()                                                                                                                                                   
        accuracy_layer = re.search(r'layers?\s\{[^}]*type:?\s"Accuracy"[^}]*\}',full_text)
        label_layer_name = re.search(r'bottom:?\s\"((?!label)\w+)\"', accuracy_layer.group(0))                                                                                 

        added_layer = 'layer {\nname: "softmax"\ntype: "Softmax"\nbottom: "bottom_name"\ntop: "softmax"\ninclude { phase: TEST }\n}'
        added_layer = added_layer.replace('bottom_name',label_layer_name.group(1))
        false_multi_view_line = re.search(r'multi_view:\s*true',full_text)
        true_multi_view_line = false_multi_view_line.group(0).replace('true','false')

        full_text = full_text.replace(added_layer, "")                                                                                                                         
        full_text = full_text.replace(false_multi_view_line.group(0), true_multi_view_line)

        updated_file = full_text                                                                                                                                               

    output = open(file_name, 'w')
    output.write(updated_file)                                                                                                                                                 
    output.close()      

if __name__ == '__main__':
    args = sys.argv                                                                                                                                                            
    if len(args)!=3:
        print('Error, the name of _train_test.prototxt file is needed. ')
    else:
        operation = args[1]
        selfMod = __import__(__name__)
        func = getattr(selfMod,operation)
                                                                                                                                                                               
        file_name = args[2]
        func(file_name)