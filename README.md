# Thesis Repository: "On the Improvement of Graph-Based Information Retrieval Models Using Machine Learning"

This repository hosts the code implementation, experiments, and results for my Integrated Master's thesis, titled **"On the Improvement of Graph-Based Information Retrieval Models Using Machine Learning"**, along with its byproduct—a paper titled **"Spectral Clustering and Query Expansion on the Graphical Set-Based Model"**. For these purposes, a Python library named **`infre`** was developed, supporting classical Information Retrieval (IR) models such as:

- **Vector Space (VS)** [Link to VS model reference]
- **Set-Based (SB)** [Link to SB model reference]
- **Graphical Set-Based (GSB)** [Link to GSB model reference]

### Overview of the `infre` Library

In this study, a new family of IR models was designed and implemented by extending the SB and GSB models. The **`infre`** library is envisioned to evolve, hosting various state-of-the-art (SOTA) IR models. The goal is to allow easy fitting to specific collections and seamless comparison across models using IR metrics like *Precision, Recall, F1-Score, AVEp, MAP, DCG, etc.*, and others for educational and research purposes.

### Repository Structure

- **`experiments/`**: This folder contains all the Python scripts (`.py` files) used for running the associated experiments. Results are exported as `.xlsx` files for further analysis.
- **`notebooks/`**: The final analysis is performed through Jupyter notebooks (with filenames containing `*-exploitation`). These notebooks load the `.xlsx` files, compare the results across models, and validate our hypothesis—demonstrating that introducing a conceptualized aspect into graph-based IR models improved precision by **x%**.

### Example Usage

- **`main.py`**: Serves as an example for instantiating, running, and evaluating the models on the *CF collection*. It demonstrates the ease of using the `infre` library for benchmarking different IR models.

### Future Vision

The future goal of the **`infre`** library is to become a comprehensive tool for IR research, facilitating the implementation and comparison of a wide range of IR models. As it evolves, it will aim to incorporate more SOTA methods and provide a flexible platform for testing new ideas in the field of Information Retrieval.

---

Feel free to clone, explore, and contribute to the project!
