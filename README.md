# bad-caviar
Zhou Fan (@Evensgn)

Text classification on hotel comments, course work 1 of PPCA 2017, ACM Class, SJTU.

## Algorithm
* `TF-IDF` + `PCA(rank = 1000)` + `SVM(rbf kernel)`: AUC = 0.94177
* `pre-trained word2vec` + `SVM(rbf kernel)`: AUC = 0.81986
