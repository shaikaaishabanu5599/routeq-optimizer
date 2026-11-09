# Security Policy

## Supported Versions

Only the current main branch and stable release tags are actively maintained and monitored for security considerations:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

---

## 🔒 Zero-Secrets & Credentials Policy

RouteQ is an open-source, reproducible quantum logistics platform designed for local classical emulation on Qiskit Aer as well as optional execution on IBM Quantum physical QPUs.

1. **No Hardcoded Secrets**: This repository contains **zero** hardcoded API keys, passwords, database credentials, or secret tokens.
2. **Qiskit IBM Runtime Credentials**: Users who choose to run circuits on IBM Quantum hardware must configure their IBM Quantum token locally using the official Qiskit secret management command:
   ```python
   from qiskit_ibm_runtime import QiskitRuntimeService
   QiskitRuntimeService.save_account(channel="ibm_quantum", token="<YOUR_TOKEN>", overwrite=True)
   ```
   Or via environment variables (`QISKIT_IBM_TOKEN`). Secrets are never committed to this repository.
3. **Environment Isolation**: A `.gitignore` rule is configured to block `.env`, `.env.local`, `.pem`, `.key`, and other sensitive extensions.

---

## 🛡️ Reporting a Vulnerability

If you discover a security vulnerability in RouteQ, please report it responsibly:

1. **Do Not Open a Public Issue**: Please do not file public GitHub issues for security vulnerabilities.
2. **Contact Team `nextgen`**: Submit a confidential advisory report through GitHub Security Advisories on this repository or email the maintainers directly.
3. **Information to Include**:
   - Description of the vulnerability and attack vector
   - Steps to reproduce or proof-of-concept script
   - Potential impact on quantum circuit synthesis, API execution, or client interfaces
   - Any suggested mitigations or patches

We will acknowledge receipt within 48 hours and work with you on a resolution before any public disclosure.
