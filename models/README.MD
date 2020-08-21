To add a new Graph Neural Network (GNN) model, create a new python file with the model name as the file name. 
In this file, declare a class for the model while using the same model name that is used for the file name.
All GNN models take in only two parameters in its constructor: config and dataset features dictionary

The class follows the standard torch nn module, containing:
1. init function, you can declare and instantiate layers required for the model using the parameters defined in the configuration file.
Only the configurations under the name corresponding with the model name will be loaded.
2. A forward function. The forward function takes in 4 arguments:
- node_feat: the input information matrix in a dense form
- adjacency_matrix: a sparse matrix representing the edge pairs
- subg_size: number of graphs in the batch
- batch_graph: the batch size
3. A loss function

See DGCNN.py for an example.