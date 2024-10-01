
# SchedGPT Dataset designed to train LLMs on JSSP

## Dataset Overview

**Dataset Name:** jssp_llm_format_120k.json
**Number of Entries:** 120,000  
**Number of Fields:** 5  

## Fields Description

1. **num_jobs**
   - **Type:** int64
   - **Number of Unique Values:** 12
   
2. **num_machines**
   - **Type:** int64
   - **Number of Unique Values:** 12
   
3. **instruction**
   - **Type:** object
   - **Number of Unique Values:** 120,000
   - **Initial description of the problem detailing the number of jobs and ma-
      chines involved.

5. **output**
   - **Type:** object
   - **Number of Unique Values:** 120,000
   - **Solution in LLM format:** 120,000

6. **matrix**
   - **Type:** object
   - **Number of Unique Values:** 120,000
   - **Input problem OR-Tool makspan and solution in Matrix format** 

   
## Usage

This dataset can be used for training LLMs for job-shop scheduling problems (JSSP). Each entry provides information about the number of jobs, the number of machines, and other relevant details formatted in natural language.

## License

This dataset is licensed under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0). For more details, see the [license description](https://creativecommons.org/licenses/by-sa/4.0/). The dataset will remain accessible for an extended period.

