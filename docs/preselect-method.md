# PreSelect 用于数据筛选挑战的详细方法

本文档旨在为数据筛选挑战赛提供一套详细的、基于 PreSelect 方法论的数据筛选方案。我们的核心目标是为四个特定的下游任务——**角色扮演 (Roleplay)**、**推理 (Reasoning)**、**函数调用 (Function Calling)** 和 **检索增强生成 (RAG)**——创建专门的、高质量的数据集。

## 核心思想

原始的 PreSelect 方法通过寻找"模型压缩率（即Loss）"与"通用下游任务表现"之间存在强相关性的数据，来筛选出对通用预训练最有效的数据。我们将此方法进行调整，将"通用下游任务表现"替换为"在挑战赛中某个特定任务（例如：角色扮演）上的表现"。通过这种方式，我们可以为每个任务训练一个专门的数据筛选器。

## 阶段一：准备工作与信号收集

本阶段的目标是收集训练 fastText 分类器所需的所有原始数据和信号。

### 步骤 1.1：创建"探针模型"集合 (Model Zoo)

-   **目标**：生成一组具有多样性的"探针"模型，用于衡量不同数据的特性。我们的目标不是将模型训练到最佳，而是使其能力产生"偏向性"，从而在不同类型的文本上给出有区分度的损失值。

-   **核心方法**：以比赛提供的 **起始模型 (Starter Model)** 为基础，使用少量数据对其进行短暂的、小规模的微调。建议进行 100-500 个优化步数即可。

-   **建议的模型与数据集列表**：

    1.  **模型0：基础模型 (Baseline Model)**
        -   **数据集**: 无，直接使用官方 **Starter Model**。
        -   **目的**: 作为零点参照，代表模型的原始状态。

    2.  **模型1：对话与角色扮演模型 (Conversational Model)**
        -   **数据集**: [Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) 或 [OpenAssistant Guanaco](https://huggingface.co/datasets/OpenAssistant/oasst1)。
        -   **目的**: 强化模型在 **Roleplay** 任务上的能力。

    3.  **模型2：代码与逻辑推理模型 (Code & Logic Model)**
        -   **数据集**: [The Stack (small)](https://huggingface.co/datasets/bigcode/the-stack-smol) 的一个子集 (例如，仅使用 Python 或 Java)。
        -   **目的**: 强化模型在 **Reasoning** 和 **Function Calling** 上的能力。

    4.  **模型3：函数调用专属模型 (Function Calling Model)**
        -   **数据集**: [Glaive Function Calling v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)。
        -   **目的**: 专门针对 **Function Calling** 任务进行优化的标杆模型。

    5.  **模型4：问答与检索模型 (QA & Retrieval Model)**
        -   **数据集**: [SQuAD](https://huggingface.co/datasets/rajpurkar/squad) 或 [Natural Questions](https://huggingface.co/datasets/google/natural_questions)。
        -   **目的**: 强化模型在 **RAG** 任务中所需的信息提取与利用能力。

    6.  **模型5：学术与科学推理模型 (Academic & Scientific Model)**
        -   **数据集**: [ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers) 的一个子集 (例如，仅使用计算机科学cs领域)。
        -   **目的**: 引入一个不同维度的能力，该模型可能在 **Reasoning** 方面表现好，但在非正式对话中表现差，从而制造出我们需要的"能力差异性"。

### 步骤 1.2：准备候选数据集 (Candidate Data Pool)

-   **目标**：创建一个具代表性的数据子集，用于后续的分析。
-   **方法**：
    1.  从比赛提供的完整 **起始数据集 (Starter Dataset)** 中，随机抽取一个规模适中的部分（例如，抽取总量的 1% 到 10%）。

### 步骤 1.3：评测下游任务表现

-   **目标**：衡量"探针模型"集合中每个模型在挑战赛指定任务上的具体表现。
-   **方法**：
    1.  使用官方提供的 **ELMB 验证集**。
    2.  对"探针模型"集合中的每一个模型，在全部四个任务（角色扮演、推理、函数调用、RAG）上进行评测。
    3.  最终会得到四组成绩，每个任务都有一组分数（例如，`Roleplay_scores = [模型1得分, 模型2得分, ...]`）。

### 步骤 1.4：计算每个文档的损失（Loss/BPC）

-   **目标**：衡量每个"探针模型"对"候选数据集"中每个文档的"压缩"程度。
-   **方法**：
    1.  对于"探针模型"集合中的每一个模型，计算它在"候选数据集"中**每一篇文档**上的训练损失（Loss）或 BPC (Bits Per Character)。
    2.  项目中的 `data_processing/bpc/main.py` 脚本可以用于此目的。
    3.  最终产出一个 `[模型数量 x 文档数量]` 大小的损失值矩阵。

## 阶段二：训练任务专属的分类器

收集完所有信号后，我们开始训练数据筛选器。

### 步骤 2.1：计算每个文档的"预测力分数"

-   **目标**：量化一个文档的损失模式（Loss Pattern）对于某个特定任务的成功有多大的预测能力。
-   **方法**：
    1.  遍历"候选数据集"中的每一篇文档。
    2.  针对四个任务中的每一个（例如，角色扮演任务）：
        -   计算以下两个向量之间的相关性（例如，皮尔逊或斯皮尔曼相关系数）：
            1.  **损失向量**：该文档在所有探针模型上的损失值列表 `[loss_model1, loss_model2, ...]`。
            2.  **任务表现向量**：所有探针模型在该特定任务（角色扮演）上的得分列表 `[roleplay_score_model1, roleplay_score_model2, ...]`。
    3.  计算出的这个相关系数，就是该文档对于这个特定任务的"预测力分数"。
    4.  完成这一步后，每篇文档都会得到四个独立的"预测力分数"，分别对应四个任务。

### 步骤 2.2：准备 fastText 训练数据

-   **目标**：为训练 fastText 模型创建带标签的训练集。
-   **方法**：
    1.  为每个任务创建一个独立的训练文件。
    2.  根据文档在特定任务上的"预测力分数"来赋予标签。例如，对于角色扮演任务，将预测力分数排名前 20% 的文档标记为 `__label__1`（高质量），其余标记为 `__label__0`。
    3.  这样，我们将得到四个独立的、符合 fastText 格式的 `.txt` 训练文件。

### 步骤 2.3：训练分类器

-   **目标**：为每个任务训练一个 fastText 模型。
-   **方法**：
    1.  使用 `data_processing/fasttext/train_fasttext.py` 脚本。
    2.  基于上一步准备好的四个带标签数据集，分别训练四个独立的模型。
    3.  最终产出四个分类器模型文件（例如, `roleplay_classifier.bin`, `reasoning_classifier.bin` 等）。

## 阶段三：数据筛选与最终模型训练

### 步骤 3.1：筛选完整数据集

-   **目标**：使用训练好的分类器，从全部起始数据中选出高质量数据。
-   **方法**：
    1.  针对四个任务，分别使用其对应的分类器，对完整的 **起始数据集 (Starter Dataset)** 进行扫描和分类。
    2.  使用类似于 `README.md` 中提供的（基于 `datatrove` 的）脚本来进行筛选。只保留被分类为 `__label__1` 的文档（或置信度高于某一阈值的文档）。
    3.  这一步会产出四个干净的、任务专属的数据集。

### 步骤 3.2：最终模型训练策略

-   **目标**：训练用于最终提交的模型。
-   **方法**：
    -   **策略A（冲击单项奖）**：
        1.  训练四个独立的模型。
        2.  每个模型都以 **起始模型** 为基础，分别在对应的任务专属数据集（例如，一个模型在筛选出的角色扮演数据上进行训练，另一个在推理数据上训练）上进行持续预训练。
    -   **策略B（冲击总冠军奖）**：
        1.  将四个筛选出来的任务数据集混合，创建一个单一的、更全面的训练集。可以考虑如何平衡不同任务的数据量。
        2.  在这个混合数据集上进行持续预训练，训练一个"全能型"模型。

### 步骤 3.3：提交

-   **目标**：打包并提交所需的产出物。
-   **方法**：
    1.  打包最终训练好的模型检查点 (checkpoint)。
    2.  打包筛选后的数据集。
    3.  打包数据筛选和模型训练的代码。
    4.  根据比赛规则进行提交。

## 重要注意事项与FAQ

### Q1: 我找到了官方发布的ELMB验证集（如 `data4elm/ELMB-FunctionCalling`），我应该用它来做什么？

**回答**: 这是一个非常重要的发现，但必须正确使用它以避免"数据泄露"。

-   **错误用法**: 在 **步骤 1.1** 中，使用这个官方验证集来微调你的"探针模型"。
-   **正确用法**: 在 **步骤 1.3** 中，将这个官方验证集作为"评测尺"，来衡量你所有"探针模型"在特定任务上的最终得分。

**为什么？** 我们的方法依赖于一个核心假设：模型在*未知数据*上的表现，和它在某些训练文档上的损失（Loss）是相关的。如果你用验证集去训练探针模型，它就会"记住"答案，其高分就不是真实能力的体现，这样计算出的"预测力分数"会是虚假的，整个方法就会失效。

### Q2: 整个流程看起来很复杂，我能否先只针对一个任务（例如 Function Calling）进行实验，来验证方法的可行性？

**回答**: 完全可以，而且这是一个非常推荐的策略！进行一次小规模的"试点实验"可以让你在投入全部资源之前，验证整个工作流是否有效，并排除潜在的技术障碍。

**针对 Function Calling 的简化流程如下：**

1.  **简化版 Model Zoo (步骤 1.1)**: 你只需要创建最少的模型来制造差异性。例如：
    *   **模型0**: 基础模型 (Starter Model)。
    *   **模型3**: 函数调用专属模型 (用 `Glaive Function Calling v2` 等外部数据训练)。
    *   (可选) **模型2**: 代码与逻辑推理模型 (用 The Stack 数据训练)，因为它也与结构化数据相关。
2.  **单一任务评测 (步骤 1.3)**: 你只需要使用官方的 `ELMB-FunctionCalling` 验证集，来评测你创建的几个探针模型的得分。
3.  **计算 Loss (步骤 1.4)**: 照常进行。
4.  **计算预测力 (步骤 2.1)**: 你只需要计算"损失向量"和"Function Calling任务表现向量"之间的相关性，为每个文档得到一个Function Calling预测力分数。
5.  **训练单个分类器 (步骤 2.2 & 2.3)**: 你将只创建一个`__label__1`/`__label__0`的训练集，并训练出一个 `function_call_classifier.bin`。
6.  **筛选与训练 (阶段三)**: 使用你训练好的分类器筛选数据，然后用筛选出的高质量Function Calling数据进行最终模型的持续预训练。

通过这个简化的流程，你可以用最小的成本来验证这套方法的有效性。 