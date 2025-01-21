
# **LLY-DML: Comprehensive CI/CD Integration**

LLY-DML is a cutting-edge Quantum Machine Learning (QML) project designed to drive innovation in hybrid quantum-classical workflows. This repository leverages modern CI/CD pipelines to ensure the highest standards of quality, automation, and scalability. Below, you'll find an overview of the project, its workflows, and how to get started.

---

## **Table of Contents**
- [About the Project](#about-the-project)
- [Key Features](#key-features)
- [Workflows Overview](#workflows-overview)
- [Getting Started](#getting-started)
- [Development Guidelines](#development-guidelines)
- [Contributing](#contributing)
- [License](#license)

---

## **About the Project**
LLY-DML aims to simplify and accelerate the development of quantum-enhanced machine learning models. By integrating robust CI/CD workflows, the repository ensures:
- Streamlined development and testing processes.
- Cross-platform compatibility for Python Wheels.
- Automated documentation generation and deployment.
- Advanced backward compatibility testing for serialized Quantum Circuits (QPY).

---

## **Key Features**
- **Comprehensive Testing**:
  - Supports Python and Rust-based components.
  - Ensures backward compatibility of QPY files.
  - Automated unit, integration, and memory safety tests.
- **Multi-Platform Builds**:
  - Supports Linux, macOS, and Windows.
  - ARM and x86_64 architecture compatibility.
- **Automated Documentation**:
  - Builds and deploys project documentation seamlessly.
- **Optimized Release Process**:
  - Builds and publishes Python artifacts (Wheels and sdist) to PyPI.

---

## **Workflows Overview**
The project includes the following GitHub Actions workflows:

1. **`coverage.yml`**:
   - Tracks test coverage for Python and Rust components.
   - Combines coverage reports using `grcov` and `coverage.py`.

2. **`docs_deploy.yml`**:
   - Automates the build and deployment of project documentation.
   - Ensures that updated documentation is always accessible.

3. **`miri.yml`**:
   - Runs memory safety checks on Rust code using Miri.
   - Detects undefined behavior and ensures Rust component integrity.

4. **`qpy.yml`**:
   - Tests backward compatibility for QPY files.
   - Validates serialized Quantum Circuits across multiple versions.

5. **`test.yml`**:
   - Runs comprehensive integration and unit tests.
   - Verifies compatibility with Python versions 3.10 and 3.13.

6. **`wheels-build.yml`**:
   - Modular workflow for building Python Wheels.
   - Configured for cross-platform builds, including ARM64 and x86_64.

7. **`wheels.yml`**:
   - Automates the full release cycle for Python artifacts.
   - Uploads distributable Wheels and sdist to PyPI.

---
