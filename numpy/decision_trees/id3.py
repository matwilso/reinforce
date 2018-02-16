import numpy as np
import pandas as pd

class Node(object):
    """
    Node for a decision tree
    """
    def __init__(self, name, attribute=None):
        self.name = name # name for debugging
        self.attribute = attribute # distinguisher
        self.label = None # label for leaf
        self.children = {} # (keys, vals) = (edges, vertices) of children nodes

    def print_tree(self):
        """
        Fairly ugly tree print routine for debugging
        """
        print("Root ", self.name, self.attribute, self.label)
        for child in self.children:
            print("{} -> {}".format(child, self.children[child].name))
        for child in self.children:
            self.children[child].print_tree()

class ID3(object):
    """
    Object for creating a Decision Tree based using ID3 algorithm

    Example usage:

    ```
    from id3 import ID3

    id3 = ID3()
    id3.fit(data, labels, data_column_names)

    predicted_label = id3.eval(test_data_point)
    ```
    """
    def __init__(self, split_mode="information", max_depth=7, log_level=0):
        self.split_mode = split_mode
        self.max_depth = max_depth
        self.log_level = log_level

    def _calc_ent(self, cat, name=None):
        """Helper function to calculate the expected entropy of vector of labels"""
        num_tot = (cat.shape[0])
        category_probs = []
        for c in set(cat):
            num_match = np.count_nonzero(cat == c)
            category_probs.append(num_match / num_tot)
            if name is not None:
                print("Num match = {}, Total = {}".format(num_match, num_tot))
        entropy = -np.sum(category_probs * np.log2(category_probs))
        return entropy
    
    def _calc_E_ent(self, cat, labels, name=None):
        """
        Helper function to calculate the expected entropy for a column of data,
        w.r.t. labels
        """
        tot_expected = 0
        for c in set(cat):
            prob_of_c = labels[cat == c].shape[0] / labels.shape[0]
            ind_ent = self._calc_ent(labels[cat == c])
            if self.log_level >= 2 and name is not None:
                print("[{} = {}]".format(name, c))
                print("Probability = {}".format(prob_of_c))
                print("Entropy = {}".format(ind_ent))
            tot_expected += prob_of_c * ind_ent
        if self.log_level >= 1 and name is not None:
            print("{} Total expected = {}".format(name, tot_expected))
        return tot_expected

    def _max_count(self, column):
        """Helper function to return most common element in categorical array"""
        (vals, counts) = np.unique(column, return_counts=True)
        return vals[np.argmax(counts)]


    def eval(self, datapoint):
        """Evaluate a single datapoint on this Decision Tree"""
        return self._eval_recur(self.root_node, datapoint)

    def _eval_recur(self, node, datapoint):
        """Recursive implementation of eval"""
        # if node is labeled, it is a root
        if node.label is not None:
            return node.label

        # get graph edge based on data's nth element
        edge = datapoint[node.attribute]
        # check for error (should not happen ever)
        if not (edge in node.children):
            raise Exception("EVAL Error, edge {} not found in node children {}".format(edge, node.children))

        next_node = node.children[edge]
        return self._eval_recur(next_node, datapoint)

    def fit(self, data, labels, names=None):
        """Build a decision tree based on data"""
        self.og_names = names
        self.og_data = data
        self.root_node = Node('root')
        self._fit_recur(self.root_node, 0, data, labels, names)

        if self.log_level >= 2:
            self.root_node.print_tree()

    def _split_IG(self, root_node, data, labels, names):
        """Return split node based on information gain"""
        root_ent = self._calc_ent(labels)
        D = data.shape[-1]
        igs = np.zeros(D, dtype=np.float64)
        for i in range(D):
            x = data[:,i]
            if names is not None:
                name = names[i]
                ee = self._calc_E_ent(x, labels, name)
            else:
                ee = self._calc_E_ent(x, labels)
            if np.isclose(ee, 0.0):
                if self.log_level:
                    print("Entropy 0 on "+root_node.name)
            igs[i] = root_ent - ee
        split = np.argmax(igs)
        return split

    def _split_MajError(self, root_node, data, labels, names):

        D = data.shape[-1]
        ave_maj_error = np.zeros(D)

        for i in range(D):
            column = data[:, i]
            maj_errors = []
            for cat in set(column):
                label_subset = labels[column == cat]
                
                majority_label = self._max_count(label_subset)
                count_majority =  np.count_nonzero(label_subset == majority_label)
                count_total = label_subset.shape[0]

                majority_error = (count_total - count_majority) / count_total
                maj_errors.append(majority_error)

            ave_maj_error[i] = np.mean(maj_errors)

        return np.argmin(ave_maj_error)



    def _fit_recur(self, root_node, depth, data, labels, names):
        """Recursive implementation of building a Decision Tree based on data"""
        for l in set(labels):
            if np.all(labels == l):
                root_node.label = l
                return root_node
        if data.shape == (0,):
            print("no data left")
            return root_node

        # identify splitting attribute
        if self.split_mode == "information":
            split = self._split_IG(root_node, data, labels, names)
        elif self.split_mode == "majority":
            split = self._split_MajError(root_node, data, labels, names)
        else:
            raise NotImplementedError("Only 'information' and 'majority' are supportd")


        og_split = np.where(self.og_names == names[split])[0][0]
        root_node.attribute = og_split
        root_node.name = self.og_names[og_split]

        # split the data and divide tree branches/leaf nodes accordingly
        split_column = data[:, split]
        
        for b in set(self.og_data[:, og_split]):
            branch_node = Node(names[split]+"_leaf")
            # map to the branch according to value of b
            root_node.children[b] = branch_node

            # split data to only match the b
            data_split = data[split_column == b]
            label_split = labels[split_column == b]

            if np.count_nonzero(data_split) == 0 or depth >= self.max_depth:
                branch_node.label = self._max_count(labels)
                continue

            data_sub = np.delete(data_split, split, axis=1)
            self._fit_recur(branch_node, depth+1, data_sub, label_split, np.delete(names, split)) 


if __name__ == '__main__':
    test_df = pd.read_csv('test.csv')
    train_df = pd.read_csv('train.csv')

    train_data = train_df.values[:,:-1]
    train_labels = train_df.values[:,-1]
    test_data = test_df.values[:,:-1]
    test_labels = test_df.values[:,-1]
    names = train_df.columns[:-1]

    for split_mode in ['information', 'majority']:
        split_table = np.zeros([7, 2]) 
        print()
        print(split_mode.upper().center(32))
        print("-"*32)
        for j in range(1,8):
            id3 = ID3(split_mode=split_mode, max_depth=j)
            id3.fit(train_data, train_labels, names)
    
            train_count = 0
            N = train_df.values.shape[0]
            for i in range(N):
                if id3.eval(train_data[i]) == train_labels[i]:
                    train_count += 1
            split_table[j-1,0] = train_acc = train_count / N
    
            test_count = 0
            M = test_df.values.shape[0]
            for i in range(M):
                if id3.eval(test_data[i]) == test_labels[i]:
                    test_count += 1
            split_table[j-1,1] = test_acc = test_count / M

        df = pd.DataFrame(split_table, columns=['Train Accuracy', 'Test Accuracy'])
        df.index += 1

        from IPython.display import display
        display(df)
