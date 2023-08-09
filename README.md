# Exploring Conformational Landscape of Cryo-EM Using Energy-Aware Pathfinding Algorithm
We introduce a novel energy-aware pathfinding algorithm designed to search for the most probable conformational changes in cryo-EM datasets. This approach seeks the shortest pathway on a graph, with edge weights defined as free-energy-like values. Unlike traditional methods that typically operate energy landscape in two or three dimensions (as seen in MEP searches), our algorithm is capable of functioning in higher dimensions. We have tested our method on both synthetic data and the real-world dataset EMPIAR-10076. 

## Setup
We developed our approach based on cryoDRGN, a well-known model for analyzing heterogeneity in cryo-EM data. Users can execute our approach in the cryoDRGN environment. Additionally, we utilize cryoDRGN to generate 3D volumes for calculating FSC which is one of our metric to evluate performance.

### Quicktest
After successfully installing cryoDRGN, users can navigate to the testing directory and run:
```{bash}
./quicktest.sh
```
This will spend a few minutes searching for the best pathway on one of our synthetic datasets (Hsp90) with a threshold set by a quantile of 0.2. The results will be saved in the testing directory.

## Notebook
The Jupyter notebook, `main.ipynb`, located in the **hsp90** directory, details our experiments. We adopted the same workflow for other datasets in our paper. Additionally, the notebook provides more information about how we compared our method to other pathfinding algorithms and the metrics we used.

## Run in Python Script
If you prefer running our approach directly as a Python script, use the command:
```{sh}
python eng_graph_search.py 
```
The inputs are:
1. **Representation coding** (e.g., Latent space from cryoDRGN).
2. **Quantile for searching** `--search-q`. If a user does not specify, the algorithm will search from 0.1 to 0.9 in increments of 0.1.
3. **Minimum and maximum zero energy ratio** `--zero-rato-l`, `--zero-ratio-h`. This range controls the shape of the energy landscape. Refer to our paper for more details. The default range is $(0.01, 0.1)$.
4. **Output directory** `-o`.
5. (Optional) Output all paths. To output all paths that reach the range of zero energy ratio, add the `--output-all` flag to the command.

 ## Synthetic Data Generation
 The workflow for generating synthetic datasets can be found at [NLRP3](https://github.com/tengyulin/synth_nlrp3.git) and [Hsp90](https://github.com/tengyulin/synth_hsp90.git).
