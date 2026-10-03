# 🏥 Clinical 30-Day Hospital Readmission Risk API

A production-ready RESTful microservice built with **FastAPI** and **Scikit-Learn** to predict 30-day hospital readmission risk based on clinical patient features. Designed for healthcare informatics workflows with built-in input validation, error handling, and automated unit test coverage.
## 🐳 Running with Docker

You can containerize and run the API in an isolated environment without setting up a local Python virtual environment.

### 1. Build the Docker Image
```bash
docker build -t readmission-api .