# PDF Chatbot with Retrieval-Augmented Generation (RAG)

This project integrates a Retrieval-Augmented Generation (RAG) framework that enables users to interact with content extracted from uploaded PDF documents. It utilizes LlamaIndex for robust document indexing and provides a user-friendly interface through Django. The application is designed to be compatible with both CPU and GPU environments, supporting varying levels of VRAM.

## Features

- **Interactive Chat with PDFs**: Directly chat with the textual content of uploaded PDF documents.
- **Django Web Interface**: A clean and intuitive web interface for easy interaction with the system.
- **Adaptive Hardware Compatibility**: Configurable for different hardware setups including options for no VRAM, low VRAM, or high VRAM environments.
- **Docker Integration**: Ensures consistent environments and easy deployment using Docker.

## Installation Instructions

Follow these steps to set up and run the project:

### Prerequisites

- Docker and Docker Compose installed on your machine.
- Basic knowledge of terminal/command prompt usage.

### Setup

1. **Clone the Repository**

    Start by cloning the repository to your local machine:

    ```bash
    git clone <repository-url>
    ```

2. **Create a Docker Network**

    This network will allow your Docker containers to communicate internally:

    ```bash
    docker network create my_shared_network
    ```

3. **Prepare the Django Application**

    Navigate to the Django web app directory and set up the database:

    ```bash
    cd web_app
    python manage.py makemigrations
    python manage.py migrate
    ```

4. **Configure VRAM Settings**

    Adjust the VRAM settings according to your hardware capabilities by editing the `rag_engine/config/dev.yaml` file:

    ```yaml
    vram: "choose from [no_vram, low_vram, high_vram]"
    ```

5. **Request Access to Llama-3.2 Model**

    You need to request access to the Llama-3.2 model for this application to function. Request access at [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).

6. **Create .env File for API Token**

    Create a `.env` file under the `rag_engine/` folder and add your Hugging Face account token:

    ```plaintext
    # rag_engine/.env
    HUGGING_FACE_HUB_TOKEN=<your_hugging_face_account_token>
    ```

7. **Download Necessary Models**

    Download the models required for the RAG engine:

    ```bash
    cd rag_engine
    sh download_models.sh
    ```

8. **Launch the Application**

    Based on your hardware, start the application using Docker Compose:

    - For **GPU** usage:

        ```bash
        docker compose --profile gpu up
        ```

    - For **CPU** usage:

        ```bash
        docker compose --profile cpu up
        ```

### Access the Chatbot Interface

Once everything is set up, access the chatbot by navigating to:

[http://127.0.0.1:8000/ai/chatbot/](http://127.0.0.1:8000/ai/chatbot/)

## Usage

Upload PDF files through the web interface and start chatting. The system will retrieve information from the PDF and generate responses based on your queries.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your features or fixes.


## Contact

For any queries or technical issues, please file an issue on the GitHub repository.

---

Enjoy interacting with your PDFs in a whole new way!
