# Security Policy

## Supported Versions

The following table lists which versions of librosa currently receive security updates.

| Version | Supported          |
| ------- | ------------------ |
| 0.10.x  | :white_check_mark: |
| < 0.10  | :x:                |

## Reporting a Vulnerability

**Important:** librosa is a pure Python library built on top of standard scientific Python packages (NumPy, SciPy, etc.). Many apparent security issues reported against librosa are actually issues in its dependencies. Before reporting, please verify that the vulnerability is specific to librosa's code and not in NumPy, SciPy, or another dependency.

If you believe you have found a security vulnerability in librosa, please report it responsibly:

1. **Do not** disclose the issue publicly (e.g., in a GitHub issue) until it has been addressed.
2. Email the details to **brian.mcfee@nyu.edu**
3. Include the following in your report:
   - A description of the vulnerability and its potential impact
   - Steps to reproduce the issue (ideally with minimal code)
   - The version(s) of librosa affected
   - Any potential mitigations you've identified

## What to Expect

- You will receive an acknowledgment within 5 business days
- We will investigate and respond with our assessment within 30 days
- If confirmed, we will work on a fix and coordinate disclosure timing with you
- We request that you allow at least 90 days before public disclosure to give us time to address the issue and allow users to update

## Security Considerations for Users

Since librosa is a Python library for audio analysis, typical security concerns include:

- **File I/O:** librosa uses third-party libraries (soundfile, audioread) to load audio files. Ensure you trust the source of any audio files you process.
- **Network operations:** librosa does not make network requests directly, but some features (e.g., downloading pre-trained models) may use standard library functions.
- **Dependencies:** Keep your dependencies updated, especially NumPy and SciPy, as these may contain security fixes.

## Learning More

For general guidance on securing Python applications, see:
- [Python Security](https://python-security.readthedocs.io/)
- [OWASP Python Security](https://owasp.org/www-project-python-security/)
