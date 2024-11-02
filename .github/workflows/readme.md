# .github/workflows/execute.json

/* Configuration Description:

- "run_test": When set to 'true', all tests located in the 'module/test' directory are executed. 
  This option enables running tests to verify code stability and ensure consistency before further steps.

- "run_training": When set to 'true', the file 'main.py' will be executed in training mode. 
  This is done by using the command 'python main.py -r', which initiates the training phase.
  This option is essential for initiating the model's training process and carrying out optimization and tuning.

- "machine_for_tests": Specifies the machine on which the tests will be executed. This option has two
  possible values:
    * "actions": Uses GitHub Actions space for testing. This is suitable for quick and simple tests
      that do not require substantial computational power.
    * "cluster": Utilizes the computing cluster for testing. This is suitable for extensive tests 
      that require higher computational power and parallel processing.


# .github/workflows/execute.json (continued)

    - "machine_for_training": Specifies the machine on which the training will be executed. This option also
      has two possible values:
        * "actions": Uses GitHub Actions space for training. This is practical if the model training
          does not require substantial computational resources.
        * "cluster": Utilizes the computing cluster for training. This is suitable for complex training
          processes that demand more resources and computational capacity.

    - "excluded_tests": This element is used to exclude specific test files from execution. Here, filenames 
      should be entered separated by commas, e.g., 'test_circuit.py, test_test.py'. Note that all listed 
      files should be located in the 'module/test' directory.

    Note: The parameters "machine_for_tests" and "machine_for_training" allow for flexible selection between 
    GitHub Actions and a computing cluster, depending on the requirements of the test or training tasks.
*/


# .github/workflows/config.json

This configuration controls version increments, and it is structured as follows:

```json
{
    "increase": {
      "stable": true,
      "dev": true,
      "main": true
    }
}

# .github/workflows/config.json

This configuration controls version increments, and it is structured as follows:

```json
{
    "increase": {
      "stable": true,
      "dev": true,
      "main": true
    }
}
Depending on which branch is set to true or false, the version is incremented accordingly:

When pushing to stable, the middle version number increases by one, and the last digit resets to zero (e.g., 1.2.0).
When pushing to main, the first version number increases by one, resetting the other numbers to zero (e.g., 2.0.0).
When pushing to dev, only the last version number increases by one (e.g., 1.2.3).
vbnet
Code kopieren

---

```markdown
# .github/workflows/gitlog.yaml

## Functionality Description

This workflow file, named **Versionsverwaltung** (Version Control), is triggered upon a push to any of the `dev`, `stable`, or `main` branches. The steps it performs are as follows:

1. **Determine Branch**: The current branch is identified to know which version component (major, minor, or patch) should be incremented based on the branch's configuration.

2. **Read Current Version**: Reads the current version from the `DML_VERSION` variable to determine the base version.

3. **Configuration Check**: The workflow reads `config.json` to check which branches (`dev`, `stable`, `main`) are set to increment their respective parts of the version.

4. **Version Update Logic**: Depending on the branch:
   - `dev` increments the last part of the version.
   - `stable` increments the middle part and resets the last.
   - `main` increments the first part and resets the middle and last parts.

5. **Update `DML_VERSION`**: If the version has changed, `DML_VERSION` is updated with the new version, reflecting the latest increment based on the branch rules.

6. **Collect Python Files**: Identifies all Python files within the repository to prepare for potential updates to their version headers.

7. **Update Python Files**: Iterates through each Python file, checking for an existing version header. If a header is found, the workflow updates the version number. If no header exists, a new one is added at the top with details such as project name, version, author, and contact.

8. **Commit and Push Changes**: Commits and pushes any updates made to files back to the repository, finalizing the version control process.

This workflow effectively automates version tracking and file management across the different branches of the project.





