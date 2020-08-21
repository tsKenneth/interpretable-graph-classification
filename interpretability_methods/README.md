To add a new interpretability method, your function must take in (following the same argument order):
method(classifier_model, config, dataset_features, GNNgraph_list, current_fold, cuda=0)

1. The trained classifier model (classifier)
2. The whole parsed configuration dicionary (config)
3. The dataset features dictionary obtained from load_data() (dataset_features)
4. The training dataset (GNNgraph_list)
5. Current fold number, needed for DeepLIFT (current_fold)
6. Whether to enable cuda, 0 for disable 1 for enable (cuda)

The function must output atleast:
1. A list of dictionaries, where the first key contains the GNNgraph, and subsequent key-value pair contains the 
attribution score resulting from applying the interpretability method when using target as its key.
Example:
[{"graph": GNNgraph1, "0":[4.3,1.2,1.1,0.9], "1": [2.3,0.5,-1.1,0.4]}, 
{"graph": GNNgraph2, "0":[3.3,1.1,1.5,1.9,0.5], "1": [-2.2,0.3,-1.0,0.6,0.1]}]

2. On the second return parameter, you can optionally output a dictionary to generate images 
, where the key refers to the title of the image and the value contain a list of GNNgraph and node_attribution scores tuple.
Example:
{"comparison_with_zero_tensor": [(GNNgraph1, [4.3,1.2,1.1,0.9]), GNNgraph2([3.3,1.1,1.5,1.9,0.5])]

3. Finally on the last return parameter, you should return the execution time taken (on average) to generate an
attribution score for one graph.

4. Add the function to the a python file of the same name as put it in this directory. Import the function in \_\_init__.py

5. In config.yml, add the name of your function under "interpretability methods", and have atleast one configuration "enabled" with a value of boolean either True or False (case sensitive)

See the implementation for DeepLIFT as example