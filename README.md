  <img src="https://github.com/user-attachments/assets/2b6d0eba-8297-4f34-8cf3-c9a261f9e17e" alt="LLY-DML Logo">
</div>

[![Python](https://img.shields.io/pypi/pyversions/lly-dml.svg)](https://badge.fury.io/py/lly-dml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Discussions](https://img.shields.io/github/discussions/LILY-QML/LLY-DML)](https://github.com/LILY-QML/LLY-DML/discussions)
[![Wiki](https://img.shields.io/badge/Documentation-Wiki-blue)](https://github.com/LILY-QML/LLY-DML/wiki)
[![Paper](https://img.shields.io/badge/LLY--DML%20Paper-PDF-blue)]()
---


# **LLY-DML: Differentiable Machine Learning**

**LLY-DML** is a core component of the [**LILY Project**](https://www.lilyqml.de), focusing on developing and optimizing quantum circuits with differentiable machine learning techniques. This project enables researchers and developers to experiment with quantum-enhanced models in a user-friendly and accessible environment.

---

## **Features**
- **Optimized Quantum Circuits**: Tools for creating and refining quantum algorithms using differentiable optimization techniques.
- **Multiple Optimizers**: Various optimization algorithms (Adam, SGD, RMSProp, etc.) for different training scenarios.
- **Cross-Training**: Training of multiple activation matrices with random selection for robust quantum state preparation.
- **Automated Reporting**: Generates PDF reports with training results and performance metrics.
- **Community Collaboration**: Open for contributions and discussions to improve and expand the platform.
- **Seamless Integration**: Available through the [LILY QML platform](https://www.lilyqml.de), providing easy access to resources and tools.

---

## **Quick Links**
- 🌐 **Website**: [LILY-QML Platform](https://www.lilyqml.de)
- 📚 **Documentation**: [LLY-DML Wiki](https://github.com/LILY-QML/LLY-DML/wiki)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/LILY-QML/LLY-DML/discussions)
- 📧 **Contact**: [info@lilyqml.de](mailto:info@lilyqml.de)



## **How to Get Started**
1. Clone the repository:
   ```bash
   git clone https://github.com/LILY-QML/LLY-DML.git
   cd LLY-DML
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   For development and testing, also install the development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
3. Run the application:
   ```bash
   python dml/main.py
   ```
4. Run the tests:
   ```bash
   python dml/test.py
   ```

For more detailed instructions, refer to the [Wiki](https://github.com/LILY-QML/LLY-DML/wiki).

## Models

LLY-DML provides pre-built models in the `models` directory:

### LLY-DML-M1

A demonstration model for quantum state classification. This model takes input matrices and classifies them to specific quantum states using the DML framework.

To use this model:

```bash
cd models/LLY-DML-M1
python start.py train  # Train the model
python start.py run    # Run the model with input matrices
```

See the [LLY-DML-M1 README](models/LLY-DML-M1/README.md) for more details.

---

## **Contributors**

### Core Team

| Role                     | Name          | Links                                                                                                                |
|--------------------------|---------------|----------------------------------------------------------------------------------------------------------------------|
| Project Lead             | Leon Kaiser   | [ORCID](https://orcid.org/0009-0000-4735-2044), [GitHub](https://github.com/xleonplayz)                              |
| Supporting Contributors  | Eileen Kühn   | [GitHub](https://github.com/eileen-kuehn), [KIT Profile](https://www-kseta.ttp.kit.edu/fellows/Eileen.Kuehn/)        |
| Supporting Contributors  | Max Kühn      | [GitHub](https://github.com/maxfischer2781)                                                                          |

### Other Contributors

| Contributor                                               | Role                                       | Contribution                                      |
|-----------------------------------------------------------|--------------------------------------------|---------------------------------------------------|
| [Clausia](https://github.com/clausia)                     | Support in Development                     | General development support                       |
| [MrGilli](https://github.com/orgs/LILY-QML/people/MrGilli) | Support in Quplexity DML Version           | [Quplexity DML Development](https://github.com/MrGilli?tab=repositories) |
| [Supercabb](https://github.com/orgs/LILY-QML/people/Supercabb) | Support in Code Development                | Codebase contributions                            |
| [Userlenn](https://github.com/userlenn)                   | Support in Code Development                | Codebase contributions                            |

---

## **Public Collaboration**

We invite everyone to contribute to LLY-DML. Here's how you can help:
- **Discussions**: Share your ideas or ask questions in our [GitHub Discussions](https://github.com/LILY-QML/LLY-DML/discussions).
- **Issues**: Report bugs or request features in the [Issues section](https://github.com/LILY-QML/LLY-DML/issues).
- **Wiki**: Explore or expand our [Wiki documentation](https://github.com/LILY-QML/LLY-DML/wiki).

---

## **License**

This project is licensed under the **MIT License**. See the [LICENSE](https://opensource.org/licenses/MIT) file for details.
