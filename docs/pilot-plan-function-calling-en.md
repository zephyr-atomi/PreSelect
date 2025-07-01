# Pilot Plan: Applying PreSelect Method for Function Calling Data Filtering

This document is a concrete, executable pilot plan aimed at end-to-end validation of the PreSelect methodology through the **Function Calling** task. Upon successful completion, this process can be extended to other tasks.

## Objective

Use minimal cost and resources to complete the entire workflow from "creating probe models" to "producing specialized datasets" to "evaluating effectiveness", and quantitatively demonstrate that our filtered data can significantly improve model performance on Function Calling tasks.

---

## Phase 1: Preparation and Signal Collection

### Step 1.1: Create Simplified "Probe Model" Collection

We don't need a complete Model Zoo; we only need to create "diversity" related to Function Calling capabilities.

- **Core Method**: Use the **Starter Model** provided by the competition as a baseline, and perform brief fine-tuning with small amounts of external data.

- **Fine-tuning Data Volume and Steps (Important!)**:
  - **Data Volume**: Use several thousand to tens of thousands of high-quality samples for each fine-tuning task, focusing on domain purity of the data.
  - **Steps**: Approximately 100-500 optimization steps is a good starting point.
  - **Risks and Trade-offs**:
    - **Insufficient Fine-tuning**: If training steps are too few or data is too little, the difference between probe models and the base model will be too small, causing their Loss values to be very similar. This will make the subsequently calculated "predictive power scores" lose discriminative power, clustering around 0.
    - **Excessive Fine-tuning**: If training is too long, the model may overfit and become a "narrow expert", losing generalization ability, which will also interfere with the signal.
  - **Recommendation**: Before large-scale BPC calculation, perform a "sanity check" on a small number (e.g., 1000) of documents first, to see if the Loss distributions of different probe models have diverged.

- **Suggested Simplified Model List**:
  1. **Model 0: Baseline Model**
     - **Source**: Directly use the official **Starter Model**.
     - **Purpose**: Serves as a capability zero-point, the benchmark for measuring performance improvement.

  2. **Model A: Code & Logic Reasoning Model**
     - **Source**: Fine-tune Starter Model using Python/Java subset from [The Stack (small)](https://huggingface.co/datasets/bigcode/the-stack-smol).
     - **Purpose**: Function Calling is essentially structured text generation, similar to code structure and logic. This model provides a "generalization-related" capability dimension.

  3. **Model B: Dedicated Function Calling Model**
     - **Source**: Fine-tune Starter Model using [Glaive Function Calling v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2).
     - **Purpose**: Establish an "expert" model for this task, whose Loss will be very sensitive to text containing function calling intent.

### Step 1.2: Prepare Candidate Dataset

- **Sample Size Strategy (Cost vs. Effectiveness Trade-off)**:
  - **Main Cost**: The real computational bottleneck is the BPC calculation in Step 1.4. Calculating BPC for millions of samples using multiple models is extremely expensive.
  - **Recommended Starting Point**: Start with **100k to 200k** random samples. This quantity is usually sufficient for training an effective fastText classifier and has relatively controllable computational cost.
  - **Validation and Iteration**: After calculating "predictive power scores" using this batch of samples, plot a histogram to observe the distribution. If the signal is strong enough (wide score distribution, even bimodal), continue; if the signal is weak (scores clustered together), consider increasing sample size to 500k or more to enhance the signal.

- **Method**: Randomly sample the recommended quantity from the complete **Starter Dataset** provided by the competition for analysis.

### Step 1.3: Evaluate Function Calling Task Performance

- **Evaluation Tool**: Use the officially released validation set [**data4elm/ELMB-FunctionCalling**](https://huggingface.co/datasets/data4elm/ELMB-FunctionCalling).
- **Method**:
  1. Evaluate the three models mentioned above (baseline model, code model, FC model).
  2. Record their respective scores to get a score vector: `FC_scores = [score_model_base, score_model_A, score_model_B]`.

### Step 1.3.5: Validate Probe Models (Critical Quality Control Point)

Before investing large-scale computational resources, we must verify that the probe models are successfully constructed. Validation requires both "macro" and "micro" checks.

- **1. Macro Check (Based on Task Performance)**:
  - **Purpose**: Ensure models have learned **different capabilities**.
  - **Operation**: Reuse results from Step 1.3, i.e., the `FC_scores` vector.
  - **Success Criteria**: The score vector **must show significant diversity**. For example, `[0.50, 0.68, 0.85]` is an ideal result, indicating model capabilities have successfully diverged. While `[0.50, 0.51, 0.52]` is a failure signal.
  - **Failure Response**: Return to Step 1.1, increase fine-tuning steps or use more effective fine-tuning data.

- **2. Micro Check (Based on Loss Distribution)**:
  - **Purpose**: Ensure models have formed **different text preferences**.
  - **Operation**: Create a manual "diagnostic set" containing various types like code, function calls, general text (about 10-20 documents), and calculate BPC only on this diagnostic set.
  - **Success Criteria**: BPC loss values must meet logical expectations. For example, the code model should have the lowest loss on code samples; the function calling model should have the lowest loss on function calling samples; on general text, specialized models should not have lower loss than (may even be higher than) the baseline model.
  - **Failure Response**: If BPC values are chaotic and illogical, it indicates fine-tuning failed to inject expected "bias". Need to re-examine the fine-tuning process and data.

### Step 1.4: Calculate Loss (Loss/BPC) for Each Document

- **Method**: For the three models mentioned above, calculate their loss (Loss) on **each document** in the "candidate dataset". This produces a loss value matrix of size `[3 x number of documents]`.
- **Script**: `data_processing/bpc/main.py`

- **What exactly is "Loss" (BPC/Perplexity)?**
  - The "loss" here specifically refers to **BPC (Bits Per Character)**, which measures a model's ability to compress a piece of text, and can be understood as the model's "perplexity" about the text. **Lower BPC values indicate better model prediction on the text, meaning the text content is more compatible with the model's "knowledge system".**
  - **Calculation Process**:
    1. The model reads documents token by token in inference mode (without weight updates).
    2. For each token, the model predicts the probability distribution of the next token based on the context.
    3. We extract the probability corresponding to the actual token and calculate the cross-entropy loss at that position through `-log(probability)`.
    4. Sum all token losses in the entire document, then divide by the total number of characters (or bytes) in the document, and perform a unit conversion (divide by `log(2)`) to get the BPC value.
  - This process essentially asks: "On average, how many bits of information do I need to encode each character in this document?"

---

## Phase 2: Training Function Calling Classifier

### Step 2.1: Calculate "Predictive Power Score"

- **Method**:
  1. Iterate through each document in the "candidate dataset".
  2. For each document, calculate the correlation (Pearson or Spearman correlation coefficient) between its **loss vector** `[loss_model_base, loss_model_A, loss_model_B]` and the **task performance vector** `FC_scores`.
  3. This correlation coefficient is the document's "predictive power score" for the Function Calling task.

### Step 2.2 & 2.3: Prepare Data and Train Classifier

- **Method**:
  1. Sort documents based on the "predictive power scores" from the previous step.
  2. Mark the top N% (e.g., top 20%) documents with the highest scores as `__label__1`, and the rest as `__label__0`, generating a fastText training file.
  3. Use `data_processing/fasttext/train_fasttext.py` to train the model, producing `function_calling_classifier.bin`.

---

## Phase 3: Filtering, Training, and Effectiveness Validation

### Step 3.1: Filter Function Calling Dataset

- **Method**: Use the trained `function_calling_classifier.bin` to filter the **entire** **Starter Dataset**, saving all documents judged as `__label__1` to form a high-quality, Function Calling-specific dataset.

### Step 3.2: Train Final Model

- **Method**: Use the official **Starter Model** as a baseline, perform continued pretraining on the Function Calling dataset filtered in the previous step to get a new model `Improved_FC_Model`.

### Step 3.3: Validate Pilot Effectiveness

- **This is the most critical step of the entire pilot plan, used to prove the method's effectiveness.**
- **Method**:
  1. Use the official validation set `data4elm/ELMB-FunctionCalling` to evaluate the new model `Improved_FC_Model`'s score, recorded as `S_improve`.
  2. Obtain the original **Starter Model**'s score on this validation set `S_base` (this score was already obtained in Step 1.3).
  3. Calculate performance improvement `S = S_improve - S_base`.

- **Success Criteria**: If `S` is a significant positive number, it proves our PreSelect method is effective, and the filtered data indeed helps improve Function Calling capabilities. 