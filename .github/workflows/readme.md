# **LLY-DML GitHub Workflows**

This directory contains the CI/CD configuration for the LLY-DML project, ensuring code quality, testing, and deployment automation.

## **Workflow Overview**

### **Test Workflow (`test.yml`)**
The main testing workflow ensures code functionality across different environments:
- Runs on every push to `main` and on pull requests
- Tests on multiple operating systems (Ubuntu, macOS, Windows)
- Supports Python versions 3.9, 3.10, and 3.11
- Generates coverage reports and uploads to Codecov

### **Building Documentation**
Documentation workflow is currently under development. It will:
- Build documentation from project docstrings
- Deploy to GitHub Pages
- Update on every release and main branch push

### **Package Distribution**
Package distribution workflow is planned to automate:
- Building Python wheels for multiple platforms
- Publishing releases to PyPI
- Versioning and tagging

## **Development Guidelines**

When modifying workflows:
1. Test changes locally when possible with tools like [`act`](https://github.com/nektos/act)
2. Keep workflows modular and focused on specific tasks
3. Avoid hard-coding credentials or sensitive information
4. Update this README when adding or changing workflows

## **Future Enhancements**

We plan to add these workflows:
- Integration testing with sample quantum circuits
- Performance benchmarking
- Security scanning
- Automated dependency updates