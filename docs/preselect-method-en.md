# PreSelect Method for Data Filtering Challenge

This document aims to provide a detailed data filtering solution based on the PreSelect methodology for the data filtering challenge. Our core objective is to create specialized, high-quality datasets for four specific downstream tasks: **Roleplay**, **Reasoning**, **Function Calling**, and **Retrieval-Augmented Generation (RAG)**.

## Core Concept

The original PreSelect method identifies data where "model compression rate (i.e., Loss)" strongly correlates with "general downstream task performance" to filter out the most effective data for general pretraining. We adapt this method by replacing "general downstream task performance" with "performance on a specific task in the challenge (e.g., roleplay)". Through this approach, we can train specialized data filters for each task.

## Phase 1: Preparation and Signal Collection

This phase aims to collect all raw data and signals needed to train fastText classifiers.

### Step 1.1: Create "Probe Model" Collection (Model Zoo)

- **Objective**: Generate a diverse set of "probe" models to measure different data characteristics. Our goal is not to train models to optimal performance, but to create "bias" in their capabilities, providing discriminative loss values on different types of text.

- **Core Method**: Using the **Starter Model** provided by the competition as a baseline, perform brief, small-scale fine-tuning with small amounts of data. Recommend 100-500 optimization steps.

- **Suggested Model and Dataset List**:

  1. **Model 0: Baseline Model**
     - **Dataset**: None, directly use the official **Starter Model**.
     - **Purpose**: Serves as a zero-point reference, representing the model's original state.

  2. **Model 1: Conversational & Roleplay Model**
     - **Dataset**: [Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) or [OpenAssistant Guanaco](https://huggingface.co/datasets/OpenAssistant/oasst1).
     - **Purpose**: Enhance model capabilities for the **Roleplay** task.

  3. **Model 2: Code & Logic Model**
     - **Dataset**: A subset of [The Stack (small)](https://huggingface.co/datasets/bigcode/the-stack-smol) (e.g., only Python or Java).
     - **Purpose**: Enhance model capabilities for **Reasoning** and **Function Calling**.

  4. **Model 3: Function Calling Dedicated Model**
     - **Dataset**: [Glaive Function Calling v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2).
     - **Purpose**: A benchmark model specifically optimized for **Function Calling** tasks.

  5. **Model 4: QA & Retrieval Model**
     - **Dataset**: [SQuAD](https://huggingface.co/datasets/rajpurkar/squad) or [Natural Questions](https://huggingface.co/datasets/google/natural_questions).
     - **Purpose**: Enhance model capabilities for information extraction and utilization needed in **RAG** tasks.

  6. **Model 5: Academic & Scientific Reasoning Model**
     - **Dataset**: A subset of [ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers) (e.g., only computer science cs field).
     - **Purpose**: Introduce a different dimension of capability. This model may perform well in **Reasoning** but poorly in informal conversations, creating the "capability diversity" we need.

### Step 1.2: Prepare Candidate Dataset (Candidate Data Pool)

- **Objective**: Create a representative data subset for subsequent analysis.
- **Method**:
  1. Randomly sample a moderately-sized portion (e.g., 1% to 10% of the total) from the complete **Starter Dataset** provided by the competition.

### Step 1.3: Evaluate Downstream Task Performance

- **Objective**: Measure the specific performance of each model in the "probe model" collection on the challenge-specified tasks.
- **Method**:
  1. Use the official **ELMB validation set**.
  2. For each model in the "probe model" collection, evaluate on all four tasks (roleplay, reasoning, function calling, RAG).
  3. This will result in four score groups, with each task having a set of scores (e.g., `Roleplay_scores = [Model1_score, Model2_score, ...]`).

### Step 1.4: Calculate Loss (Loss/BPC) for Each Document

- **Objective**: Measure how well each "probe model" "compresses" each document in the "candidate dataset".
- **Method**:
  1. For each model in the "probe model" collection, calculate its training loss (Loss) or BPC (Bits Per Character) on **every document** in the "candidate dataset".
  2. The `data_processing/bpc/main.py` script can be used for this purpose.
  3. This produces a loss value matrix of size `[number of models x number of documents]`.

## Phase 2: Training Task-Specific Classifiers

After collecting all signals, we begin training data filters.

### Step 2.1: Calculate "Predictive Power Score" for Each Document

- **Objective**: Quantify how much predictive power a document's loss pattern has for success in a specific task.
- **Method**:
  1. Iterate through each document in the "candidate dataset".
  2. For each of the four tasks (e.g., roleplay task):
     - Calculate the correlation (e.g., Pearson or Spearman correlation coefficient) between the following two vectors:
       1. **Loss Vector**: The document's loss values across all probe models `[loss_model1, loss_model2, ...]`.
       2. **Task Performance Vector**: All probe models' scores on this specific task (roleplay) `[roleplay_score_model1, roleplay_score_model2, ...]`.
  3. The calculated correlation coefficient is the document's "predictive power score" for this specific task.
  4. After completing this step, each document will have four independent "predictive power scores", corresponding to the four tasks.

### Step 2.2: Prepare fastText Training Data

- **Objective**: Create labeled training sets for training fastText models.
- **Method**:
  1. Create an independent training file for each task.
  2. Assign labels based on the document's "predictive power score" for the specific task. For example, for the roleplay task, label the top 20% of documents by predictive power score as `__label__1` (high quality), and the rest as `__label__0`.
  3. This results in four independent `.txt` training files in fastText format.

### Step 2.3: Train Classifiers

- **Objective**: Train a fastText model for each task.
- **Method**:
  1. Use the `data_processing/fasttext/train_fasttext.py` script.
  2. Based on the four labeled datasets prepared in the previous step, train four independent models separately.
  3. This produces four classifier model files (e.g., `roleplay_classifier.bin`, `reasoning_classifier.bin`, etc.).

## Phase 3: Data Filtering and Final Model Training

### Step 3.1: Filter Complete Dataset

- **Objective**: Use the trained classifiers to select high-quality data from all starter data.
- **Method**:
  1. For the four tasks, use their corresponding classifiers to scan and classify the complete **Starter Dataset**.
  2. Use scripts similar to those provided in `README.md` (based on `datatrove`) for filtering. Only keep documents classified as `__label__1` (or documents with confidence above a certain threshold).
  3. This step produces four clean, task-specific datasets.

### Step 3.2: Final Model Training Strategy

- **Objective**: Train models for final submission.
- **Method**:
  - **Strategy A (Target Individual Awards)**:
    1. Train four independent models.
    2. Each model is based on the **Starter Model** and undergoes continued pretraining on the corresponding task-specific dataset (e.g., one model trains on filtered roleplay data, another on reasoning data).
  - **Strategy B (Target Grand Prize)**:
    1. Mix the four filtered task datasets to create a single, more comprehensive training set. Consider how to balance data amounts for different tasks.
    2. Perform continued pretraining on this mixed dataset to train a "versatile" model.

### Step 3.3: Submission

- **Objective**: Package and submit required deliverables.
- **Method**:
  1. Package final trained model checkpoints.
  2. Package filtered datasets.
  3. Package data filtering and model training code.
  4. Submit according to competition rules.

## Important Notes & FAQ

### Q1: I found the official ELMB validation set (like `data4elm/ELMB-FunctionCalling`). What should I use it for?

**Answer**: This is a very important discovery, but it must be used correctly to avoid "data leakage".

- **Wrong Usage**: In **Step 1.1**, using this official validation set to fine-tune your "probe models".
- **Correct Usage**: In **Step 1.3**, using this official validation set as an "evaluation ruler" to measure all your "probe models'" final scores on specific tasks.

**Why?** Our method relies on a core assumption: a model's performance on *unknown data* correlates with its loss (Loss) on certain training documents. If you use the validation set to train probe models, they will "remember" the answers, and their high scores won't reflect true capabilities. The calculated "predictive power scores" would be false, making the entire method ineffective.

### Q2: The entire process seems complex. Can I first experiment with just one task (e.g., Function Calling) to verify the method's feasibility?

**Answer**: Absolutely, and this is a highly recommended strategy! Conducting a small-scale "pilot experiment" allows you to verify whether the entire workflow is effective before investing all resources, and eliminate potential technical obstacles.

**Simplified Process for Function Calling:**

1. **Simplified Model Zoo (Step 1.1)**: You only need to create minimal models to generate diversity. For example:
   - **Model 0**: Baseline model (Starter Model).
   - **Model 3**: Function calling dedicated model (trained with `Glaive Function Calling v2` and other external data).
   - (Optional) **Model 2**: Code & logic reasoning model (trained with The Stack data), as it's also related to structured data.
2. **Single Task Evaluation (Step 1.3)**: You only need to use the official `ELMB-FunctionCalling` validation set to evaluate the scores of your created probe models.
3. **Calculate Loss (Step 1.4)**: Proceed as usual.
4. **Calculate Predictive Power (Step 2.1)**: You only need to calculate the correlation between "loss vector" and "Function Calling task performance vector" to get a Function Calling predictive power score for each document.
5. **Train Single Classifier (Steps 2.2 & 2.3)**: You will only create one `__label__1`/`__label__0` training set and train a `function_call_classifier.bin`.
6. **Filter and Train (Phase 3)**: Use your trained classifier to filter data, then use the filtered high-quality Function Calling data for final model continued pretraining.

Through this simplified process, you can verify the effectiveness of this method with minimal cost. 