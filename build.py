# FROM http://cgregistry.capgroup.com/python:3.10.13-slim-bullseye as base

# # CG pip proxies
# ARG PIP_INDEX=https://cgrepo.capgroup.com/repository/cgpypi/pypi
# ARG PIP_INDEX_URL=https://cgrepo.capgroup.com/repository/cgpypi/simple
# ARG PIP_TRUSTED_HOST=http://cgrepo.capgroup.com

# # Install system dependencies
# RUN apt-get update && \
#     apt-get install -y libpq-dev gcc

# # Create and set working directory
# WORKDIR /app

# # Copy requirements first to leverage Docker cache
# COPY requirements.txt .

# # Install Python packages
# RUN pip install --trusted-host $PIP_TRUSTED_HOST --index $PIP_INDEX  --index-url $PIP_INDEX_URL --upgrade pip && \
#     pip install --trusted-host $PIP_TRUSTED_HOST --index $PIP_INDEX  --index-url $PIP_INDEX_URL -r requirements.txt

# # Copy application code and directories
# COPY app/ ./app/
# COPY ./pdfs_qa ./pdfs_qa
# COPY ./pdf_table ./pdf_table

# EXPOSE 8501

# CMD ["streamlit", "run", "streamlit_app.py"]