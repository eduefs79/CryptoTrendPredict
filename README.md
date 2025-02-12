#CryptoTrendPredict

This project is designed to predict the next day's Bitcoin closing price using various technical indicators, including RSI, MACD, and Bollinger Bands.

## Requirements

Before you start, make sure you have the following installed on your local machine:

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Setup Instructions

### Step 1: Clone the repository


```bash

git clone https://github.com/eduefs79/CryptoTrendPredict.git
cd CryptoTrendPredict
```
### Step 2: Docker Setup
Ensure that Docker and Docker Compose are installed on your local machine. You can verify this by running:
```bash
docker --version
docker-compose --version
```
If Docker is not installed, follow the installation instructions linked above.
### Step 3: Adjust the docker-compose.yml
Before running Docker containers, you might need to adjust some settings in the docker-compose.yml file. Here are some things to check:
•	Volumes: Ensure that the volumes are correctly mapped to your local directories if you're working with local data or files.
•	Ports: Make sure the exposed ports do not conflict with other services running on your machine.
•	Environment Variables: Some environment variables may need to be configured for your local environment. Check the .env.example file and create a .env file if needed.
### Step 4: Database Setup
Make sure the MySQL user has been created for your application. You can follow the instructions in this guide to set up your database: [Create a Database in MySQL](https://docs.bitnami.com/general/infrastructure/mysql/configuration/create-database-mysql/).
### Step 5: Build and Start the Docker Containers
Once the docker-compose.yml file is adjusted, you can build and start the containers:
docker-compose up --build
This will build the Docker containers and start the services defined in the docker-compose.yml file.
### Step 6: Access the Application
After the containers are up and running, you can access your application locally (adjust based on your docker-compose.yml setup):
•	Visit [http://localhost:8000](http://localhost:8000) or the appropriate local URL.
### Step 7: Stopping the Containers
To stop the Docker containers when you're done, run:
docker-compose down
This will stop and remove the containers.

## License

This project is licensed under the MIT License - see the [LICENSE](https://choosealicense.com/licenses/mit/) file for details.


</span>


