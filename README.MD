\# Loan Risk Prediction API



A comprehensive machine learning solution for loan eligibility prediction and risk assessment, built with object-oriented principles and deployed as a RESTful API using FastAPI.



\## 🚀 Features



\- \*\*Dual Model Architecture\*\*: Classification for loan approval and regression for risk scoring

\- \*\*OOP Design\*\*: Follows SOLID principles with abstract base classes and inheritance

\- \*\*RESTful API\*\*: Complete CRUD operations for loan applicants

\- \*\*Real-time Predictions\*\*: Instant loan approval decisions and risk assessments

\- \*\*Data Processing\*\*: Automated data cleaning, feature engineering, and scaling

\- \*\*Model Persistence\*\*: Trained models saved using joblib for deployment



\## 🏗️ Architecture



\### Machine Learning Pipeline



The project implements a modular ML pipeline with the following components:



```

LoadSetDataset → PreprocessDataset → SplitDataset → Model (Abstract)

&nbsp;                                                       ├── ClassificationModel

&nbsp;                                                       └── RegressionModel

```



\#### Key Classes:



\- \*\*`LoadSetDataset`\*\*: Handles dataset download and loading from Kaggle

\- \*\*`PreprocessDataset`\*\*: Data cleaning, null handling, and size management

\- \*\*`SplitDataset`\*\*: Feature engineering and train-test splitting

\- \*\*`Model`\*\* (Abstract): Base class defining the ML model interface

\- \*\*`ClassificationModel`\*\*: Random Forest classifier for loan approval

\- \*\*`RegressionModel`\*\*: Random Forest regressor for risk scoring



\### API Architecture



\- \*\*FastAPI Framework\*\*: High-performance async API

\- \*\*Pydantic Models\*\*: Data validation and serialization

\- \*\*In-memory Storage\*\*: Simple dictionary-based storage (suitable for demo)

\- \*\*Error Handling\*\*: Comprehensive logging and exception management



\## 📊 Models



\### Classification Model

\- \*\*Purpose\*\*: Predicts loan approval (approved/rejected)

\- \*\*Algorithm\*\*: Random Forest Classifier

\- \*\*Features\*\*: Age, income, credit score, gender, loan purpose

\- \*\*Output\*\*: Binary classification (0/1)



\### Regression Model  

\- \*\*Purpose\*\*: Estimates default risk score

\- \*\*Algorithm\*\*: Random Forest Regressor

\- \*\*Features\*\*: Same as classification model

\- \*\*Output\*\*: Continuous risk score



\## 🛠️ Installation



\### Prerequisites

\- Python 3.8+

\- pip package manager



\### Setup



1\. \*\*Clone the repository\*\*

&nbsp;  ```bash

&nbsp;  git clone https://github.com/yourusername/loan-risk-prediction-api.git

&nbsp;  cd loan-risk-prediction-api

&nbsp;  ```



2\. \*\*Create virtual environment\*\*

&nbsp;  ```bash

&nbsp;  python -m venv venv

&nbsp;  source venv/bin/activate  # On Windows: venv\\Scripts\\activate

&nbsp;  ```



3\. \*\*Install dependencies\*\*

&nbsp;  ```bash

&nbsp;  pip install -r requirements.txt

&nbsp;  ```



4\. \*\*Train the model\*\* (Run the Jupyter notebook or training script)

&nbsp;  ```bash

&nbsp;  python train\_model.py

&nbsp;  ```



5\. \*\*Start the API server\*\*

&nbsp;  ```bash

&nbsp;  uvicorn main:app --reload --host 0.0.0.0 --port 8000

&nbsp;  ```



\## 📋 Requirements



```txt

fastapi==0.104.1

uvicorn==0.24.0

pandas==2.1.3

scikit-learn==1.3.2

joblib==1.3.2

pydantic==2.5.0

polars==0.19.19

kagglehub==0.2.5

numpy==1.24.3

```



\## 🔧 API Usage



\### Base URL

```

http://localhost:8000

```



\### Endpoints



\#### 1. Create Loan Applicant

```http

POST /applicants/

```



\*\*Request Body:\*\*

```json

{

&nbsp; "age": 30,

&nbsp; "income": 50000.0,

&nbsp; "credit\_score": 700.0,

&nbsp; "loan\_amount": 10000.0,

&nbsp; "gender": "Male",

&nbsp; "purpose": "Home"

}

```



\*\*Response:\*\*

```json

{

&nbsp; "applicant\_id": 1,

&nbsp; "approved": true,

&nbsp; "default\_risk\_score": 0.15,

&nbsp; "input\_data": {

&nbsp;   "age": 30,

&nbsp;   "income": 50000.0,

&nbsp;   "credit\_score": 700.0,

&nbsp;   "loan\_amount": 10000.0,

&nbsp;   "gender": "Male",

&nbsp;   "purpose": "Home"

&nbsp; }

}

```



\#### 2. Get Applicant Details

```http

GET /applicants/{applicant\_id}

```



\#### 3. Update Applicant

```http

PUT /applicants/{applicant\_id}

```



\#### 4. Delete Applicant

```http

DELETE /applicants/{applicant\_id}

```



\#### 5. Health Check

```http

GET /health

```



\### Example Usage with cURL



```bash

\# Create a new applicant

curl -X POST "http://localhost:8000/applicants/" \\

&nbsp; -H "Content-Type: application/json" \\

&nbsp; -d '{

&nbsp;   "age": 35,

&nbsp;   "income": 75000.0,

&nbsp;   "credit\_score": 750.0,

&nbsp;   "loan\_amount": 25000.0,

&nbsp;   "gender": "Female",

&nbsp;   "purpose": "Business"

&nbsp; }'



\# Get applicant details

curl -X GET "http://localhost:8000/applicants/1"



\# Health check

curl -X GET "http://localhost:8000/health"

```



\## 📁 Project Structure



```

loan-risk-prediction-api/

├── notebooks/

│   └── OOPsBasedLoanPredClassif+Regg.ipynb

├── src/

│   ├── data\_loader.py

│   ├── preprocessor.py

│   ├── models.py

│   └── utils.py

├── api/

│   ├── main.py

│   ├── schemas.py

│   └── crud.py

├── tests/

│   ├── test\_api.py

│   └── test\_models.py

├── requirements.txt

├── .gitignore

├── README.md

└── train\_model.py

```



\## 🧪 Testing



\### API Testing

```bash

\# Install test dependencies

pip install pytest httpx



\# Run tests

pytest tests/

```



\### Manual Testing

Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI).



\## 🏛️ Design Principles



This project follows SOLID principles:



\- \*\*Single Responsibility\*\*: Each class has one clear responsibility

\- \*\*Open/Closed\*\*: Models can be extended without modifying existing code

\- \*\*Liskov Substitution\*\*: Concrete models can replace the abstract Model class

\- \*\*Interface Segregation\*\*: Clear, focused interfaces for each component

\- \*\*Dependency Inversion\*\*: High-level modules don't depend on low-level details



\## 📈 Model Performance



The models are trained on loan default data with the following characteristics:

\- Dataset size: 1,247 samples (configurable)

\- Features: Age, Income, Credit Score, Gender, Loan Purpose

\- Train/Test split: 80/20

\- Evaluation metrics: Classification Report (Classification), RMSE (Regression)



\## 🚀 Deployment



\### Docker Deployment

```dockerfile

FROM python:3.9-slim



WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt



COPY . .

CMD \["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

```



\### Production Considerations

\- Replace in-memory storage with a proper database (PostgreSQL, MongoDB)

\- Implement authentication and authorization

\- Add rate limiting and request validation

\- Set up monitoring and logging

\- Use environment variables for configuration

\- Implement model versioning and A/B testing



\## 🤝 Contributing



1\. Fork the repository

2\. Create a feature branch (`git checkout -b feature/AmazingFeature`)

3\. Commit your changes (`git commit -m 'Add some AmazingFeature'`)

4\. Push to the branch (`git push origin feature/AmazingFeature`)

5\. Open a Pull Request



\## 📄 License



This project is licensed under the MIT License - see the \[LICENSE](LICENSE) file for details.



\## 👥 Authors



\- \*\*Your Name\*\* - \*Initial work\* - \[YourGitHub](https://github.com/yourusername)



\## 🙏 Acknowledgments



\- Kaggle for the loan default dataset

\- FastAPI community for excellent documentation

\- Scikit-learn team for robust ML algorithms

