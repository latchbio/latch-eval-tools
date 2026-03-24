CRITICAL INSTRUCTIONS:
1. Do NOT wrap your code in try/except blocks. Let errors propagate so you can see them and fix them in subsequent steps.
2. You must write `/workspace/eval_answer.json` BEFORE issuing the completion signal.
3. Correct order: Perform analysis -> Write eval_answer.json -> Submit with a standalone `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` command.
4. Do not print the completion marker from inside scripts and do not combine it with other commands.

The file `/workspace/eval_answer.json` should contain ONLY the JSON object with the required fields

Example `/workspace/eval_answer.json`:
{
  "field1": value1,
  "field2": value2
}
Common biological analysis packages you might need such as numpy, pandas, scipy,h5py,scanpy,anndata etc have been pre-installed. Full build environment context can be found at `/root`
