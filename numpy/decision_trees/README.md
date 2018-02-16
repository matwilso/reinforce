# Decision tree

## ID3 Algorithm

This was just from a school assignment, and I haven't cleaned it up at all.

But this is an implementation of the ID3 algorithm to create a decision tree.

My implementation assumes you have a table of data with feature columns and a
single label column, and that everything is categorical.

Vanilla ID3 calculates information gain of each column and splits on the column
that has the highest information, and that becomes the branch.  See the notebook
for visualizations.  It then goes through each branch and computes the columns
to split on (the first column split is no longer available).  It is recursive.
