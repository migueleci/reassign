# hieRarchical multi-labEl clAsSification to diScover mIssinG aNnotations (REASSIGN)

### by Miguel Romero, Felipe Kenji Nakano, Jorge Finke, Camilo Rocha, and Celine Vens

## Instructions

 1. Execute "compare_data.py" to process data and create the datasets required by the method. New folders will be created within the data folder, each one of these contains the data for one sub-hierarchy of Gene Ontology and a subgraph of the gene co-expression network for rice (*Oryza sativa Japonia*).

 2. Execute "reassign.py" to apply the method for each dataset (independently). The method will create the folder "pred" where the results will be stored. A subfolder will be created for each sub-hierarchy and will contains five files:

    - "precision.csv" resumes the predictive performance of the method (measured using precision) for different values of the number of paths (and pairwise associations between genes and functions) to be selected.

    - "precision.pdf" illustrates the predictive performance of the method in a plot containing the three variations of the method (average, sum and minimum).

    - "top_{mean,sum,min}.csv" contains the selected associations for each variation of the method. Each file contain the gene and function identifier, followed by the probability of association, the probability computed for the path containing the association and a boolean value that indicates if the association is present in the newer version of the database.

## Reference

Romero, M., Nakano, F. K., Finke, J., Rocha, C., & Vens, C. (2022). Hierarchy exploitation to detect missing annotations on hierarchical multi-label classification. arXiv preprint arXiv:2207.06237.

[https://doi.org/10.48550/arxiv.2207.06237](https://doi.org/10.48550/arxiv.2207.06237)
