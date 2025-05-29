
# Starjob Dataset designed to train LLMs on JSSP

[![Hugging Face](https://img.shields.io/badge/HuggingFace-Dataset-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/henri24/Starjob)

Dataset is available at [Hugging Face](https://huggingface.co/datasets/henri24/Starjob)



## Dataset Overview

**Dataset Name:** starjob130k.json
**Number of Entries:** 130,000  
**Number of Fields:** 5  

## Fields Description

1. **num_jobs**
   - **Type:** int64
   - **Number of Unique Values:** 16
   
2. **num_machines**
   - **Type:** int64
   - **Number of Unique Values:** 16
   
3. **instruction**
   - **Type:** object
   - **Number of Unique Values:** 130,000
   - **Initial description of the problem detailing the number of jobs and machines involved.**
     
4. **input**
   - **Type:** object
   - **Number of Unique Values:** 130,000
   - **Description of the problem in LLM format**

5. **output**
   - **Type:** object
   - **Number of Unique Values:** 130,000
   - **Solution in LLM format:** 130,000

6. **matrix**
   - **Type:** object
   - **Number of Unique Values:** 130,000
   - **Input problem OR-Tool makspan and solution in Matrix format** 

   
## Usage

This dataset can be used for training LLMs for job-shop scheduling problems (JSSP). Each entry provides information about the number of jobs, the number of machines, and other relevant details formatted in natural language.


# Setting Up Your Python Environment

Follow these instructions to create a virtual environment and install the necessary libraries.

## Step 1: Create a Virtual Environment

```bash
python3 -m venv llm_env
```

Activate the Virtual Environment
After creating the virtual environment, activate it using the following command:

On Windows
```bash
.\llm_env\Scripts\activate
```

On macOS and Linux
```bash
source llm_env/bin/activate
```

# Install the Required Libraries
```bash
pip install -r requirements.txt
```

# Training
Make sure to put dataset.json under data directory

```bash
python train_llama_3.py
```

## License

This dataset is licensed under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0). For more details, see the [license description](https://creativecommons.org/licenses/by-sa/4.0/). The dataset will remain accessible for an extended period.

