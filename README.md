AI-Powered Knowledge Graph to Manim Animation Automation System
Project Overview
The AI-Powered Knowledge Graph to Manim Animation Automation System is a comprehensive platform designed to automatically generate educational animated videos from natural language queries. By leveraging a knowledge graph, AI content generation, and the Manim animation engine, this system transforms complex educational topics into clear and engaging visual content1111. The system is built using a microservices architecture, ensuring scalability, maintainability, and fault isolation22222.






Core Features

Knowledge Graph (Neo4j): Stores educational content in an interconnected format, allowing for semantic queries and relationship discovery across various domains like Data Structures, GIS, and Space Technology3333333.





AI-Powered Script Generation (OpenAI GPT-4): Integrates with OpenAI GPT-4 to create structured, pedagogically sound educational scripts and slide content based on the knowledge graph data4444.




Manim Animation Engine: Automatically generates precise and high-quality educational visualizations by converting structured slide content into Python code for the Manim animation library555.




Microservices Architecture: The system uses a microservices pattern with components like an API Gateway, Knowledge Graph Service, AI Content Service, and Manim Render Service, each independently deployable and scalable6666666.






End-to-End Automation: The entire pipeline, from a student's natural language query to the final video delivery, is fully automated, ensuring efficient content creation7777.



System Architecture
The system is designed with a microservices architecture to ensure clear separation of concerns and scalable design principles8888. The main components communicate via an API Gateway and asynchronous message queues9999.






Component Breakdown

User Interface: A web-based front end for students to submit queries and view generated videos10.



API Gateway: The single entry point for all requests, orchestrating communication between the different microservices11111111.




Knowledge Graph Service: Manages the Neo4j database, handling queries and updates to the educational content12121212.




Content Extract: Processes unstructured data like books to populate the knowledge graph with concepts and relationships13.



AI Processing: Generates educational scripts using OpenAI GPT-4 based on content retrieved from the knowledge graph14141414.




Manim Engine: Executes the generated Python code to render the final animated video files15.



Storage System: Stores the final MP4 video files, scripts, and other metadata for efficient retrieval16.


Installation & Setup
Follow these steps to set up the system locally.

Prerequisites
Python 3.8+

Docker & Docker Compose

Neo4j Desktop or Docker Image

OpenAI API Key

Steps
Clone the Repository:

Bash

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Set up Environment Variables:

Create a .env file in the root directory with the following variables:

OPENAI_API_KEY=your_openai_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
Run with Docker Compose:

Bash

docker-compose up --build
This command will build and run all the microservices, including the Neo4j database instance.

Usage
Access the web interface at http://localhost:3000 (or the configured port).

Enter a query related to an educational topic (e.g., "Explain Binary Search Trees with an animation" or "What is the Big O notation for Quick Sort?").

The system will process the query, generate the script, and render the animation.

Once completed, the animated video will be available for viewing on the interface.

Contributing
We welcome contributions! Please see our CONTRIBUTING.md for guidelines on how to submit pull requests, report bugs, and suggest new features.

License
This project is licensed under the MIT License. See the LICENSE file for more details.
