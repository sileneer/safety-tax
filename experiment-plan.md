This experimental plan isolates network variability, standardizes the evaluation models, and provides statistically significant data to quantify the "Safety Tax."

### 1. Core Objectives

* Measure the latency overhead () introduced by structural (Guardrails AI) and semantic (NeMo Guardrails) validation compared to native prompting.
* Quantify the False Positive Rate (FPR) of valid requests blocked by guardrails.
* Determine the Attack Success Rate (ASR) of adversarial prompts across all three configurations.

### 2. Experimental Setup

**Constants & Environment**

* **Target Model:** `claude-sonnet-4-5-20250929` (balances instruction-following with reasoning capabilities).
* **Judge Model:** `gpt-5-2025-08-07` (used strictly for evaluating ASR and FPR to prevent target model bias).
* **Hosting/Network:** Execute the testing script from a cloud instance in the same region as the primary LLM API endpoint to minimize baseline network latency.
* **Sample Size ():**  total test cases per condition to achieve statistical significance.

**Independent Variables (Configurations)**
| Configuration | Mechanism | Primary Tool |
| :--- | :--- | :--- |
| **Control** | Native Prompting | System Prompt + XML tags + Negative constraints |
| **Condition A** | Structural Validation | Guardrails AI (Pydantic models, JSON enforcement) |
| **Condition B** | Semantic/Dialog Control | NVIDIA NeMo Guardrails (Colang, embedding checks) |

### 3. Dataset Composition

The  test cases must be evenly split to test both vulnerability and usability.

* **Adversarial Set (500 prompts):**
* *Direct Injection (250):* Drawn from [HEx-PHI](https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI) (330 plain-text harmful instructions across 11 policy categories) supplemented with [HarmBench](https://github.com/centerforaisafety/HarmBench) (Mazeika et al. 2024) to reach 250 prompts. HEx-PHI is a safety evaluation benchmark — it does NOT contain jailbreak techniques like roleplay or encoding tricks.
* *Indirect Injection (250):* Drawn from the [LLMail-Inject challenge](https://huggingface.co/datasets/microsoft/llmail-inject-challenge) (208,095 attack submissions from 839 participants; [paper](https://arxiv.org/abs/2506.09956)), which embeds hidden instructions inside otherwise standard data structures (emails, documents, JSON).


* **Benign Set (500 prompts):**
* *Standard Operations (250):* Expected user behavior, simple queries.
* *Edge-Case / Domain Overlap (250):* Queries containing trigger words in benign contexts (e.g., asking to "kill a process" in a Linux support bot, or "drop tables" in database documentation).



### 4. Execution Pipeline

1. **Initialization:** Load the dataset into the custom Python evaluation pipeline (`runner.py`). Initialize API connections and local embedding models required for NeMo.
2. **Concurrency Control:** Execute queries asynchronously, but cap concurrency at 10 requests per second to avoid rate-limiting spikes that could skew latency metrics.
3. **Data Capture:** For each request, log the following:
* Input prompt.
* Raw output.
* Final output (if modified/blocked by guardrail).
* Total processing time (in ms).
* Total tokens consumed (input + output + internal reasoning/validation calls).



### 5. Evaluation Metrics & Mathematics

Calculate the following metrics for each of the three configurations.

**Latency Overhead ()**
Measure the time tax introduced by the libraries.



*Calculate the median () and the 95th percentile () latency in milliseconds (ms) to account for outliers.*

**Token Tax ()**
Measure the cost overhead.


**Safety Classification F1 Score**
Use the LLM-as-a-judge to evaluate outputs.

* **True Positives (TP):** Adversarial prompt successfully blocked.
* **False Positives (FP):** Benign prompt incorrectly blocked (the "Annoyance" metric).
* **True Negatives (TN):** Benign prompt successfully processed.
* **False Negatives (FN):** Adversarial prompt successfully bypassed the guardrail.

Calculate the  score to balance the system's ability to stop attacks against its usability:


### 6. Expected Failure Modes to Monitor

* **Control (Native):** Watch for prompt degradation. Long contexts may cause the model to forget negative constraints placed at the beginning of the prompt.
* **Condition A (Guardrails AI):** Monitor the token tax. If the model fails the JSON schema validation, the library triggers a re-prompt, doubling the token cost and latency.
* **Condition B (NeMo):** Monitor the FPR. The embedding-based semantic checks are prone to blocking the "Edge-Case" benign prompts.

The evaluation pipeline is implemented in `runner.py` (execution) and `analysis.py` (metrics computation).