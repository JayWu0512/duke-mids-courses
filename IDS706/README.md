[![Python Template for IDS706](https://github.com/JayWu0512/duke-mids-courses/actions/workflows/ids706-ci.yml/badge.svg)](https://github.com/JayWu0512/duke-mids-courses/actions/workflows/ids706-ci.yml)

# Duke IDS 706 – Data Engineering Systems

This repository contains coursework, projects, and experiments for **IDS 706: Data Engineering Systems**, a core course in the Duke Master in Interdisciplinary Data Science (MIDS) program.

The course provides a **hands-on introduction to modern data engineering practices**, covering topics such as:

- Data modeling and pipelines
- SQL and relational databases
- Cloud-based data platforms
- Distributed systems and big data frameworks
- Data engineering for machine learning and analytics

The goal of this repo is to **document learning, share reproducible code, and track progress** throughout the course, while building practical skills in managing, transforming, and deploying data systems at scale.

## Setup Instructions

To get started, clone this repository and set up your environment:

```bash
# Clone the repository
git clone https://github.com/JayWu0512/duke-mids-courses.git
cd duke-mids-courses

# Install dependencies
make install
```

## Usage Examples

Use the provided **Makefile** to manage common tasks:

```bash
# Install dependencies
make install

# Format code with black
make format

# Lint code with flake8
make lint

# Run tests with pytest + coverage
make test

# Clean cache and temporary files
make clean

# Run everything (install → format → lint → test)
make all
```
