# Production AI Engineering

Learn best practices, tips, and tricks to build production-ready AI applications from beginner to advanced. The code and explanations in this series are organized in a folder structure for easier navigation

> **Note:** This is not an AI productivity series.

---

## About This Series

- Production-ready code with robust error handling
- Cost tracking and optimization strategies
- Comprehensive documentation
- Real-world examples and use cases
- Deployment considerations

---

## Project Timeline

- [**Day 1:** Production LLM Client with Retry Logic & Cost Tracking](./day-01-production-llm-client/) | [Blog Post](https://blog.chapainaashish.com.np/llm-client-with-retry-logic-and-cost-tracking)

---

## Tech Stack

- **Language:** Python 3.10+  
- **LLM API:** OpenAI  
- **Vector Databases:** FAISS, ChromaDB, Pinecone, pgvector  
- **Frameworks:** LangChain, FastAPI, React  
- **Tools:** tiktoken, Pydantic, Redis  

---

## Quick Start

### Prerequisites

- Python 3.10+  
- OpenAI API key ([Get one here](https://platform.openai.com/signup))  
- Git  

### Installation

```bash
# Clone the repository
git clone https://github.com/chapainaashish/30-days-of-ai-engineering.git
cd 30-days-of-ai-engineering

# Navigate to a specific day
cd day-01-production-llm-client

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run code
python src/client.py
````

---

## Learning Resources

Each day's project includes:

* **Detailed README** – Technical explanations and setup
* **Blog Post** – In-depth tutorial on Hashnode
* **Working Code** – Production-ready implementations

---

## Contributing

Found a bug or want to improve something? Contributions are welcome!

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/improvement`
3. Commit your changes: `git commit -am 'Add some improvement'`
4. Push to the branch: `git push origin feature/improvement`
5. Open a Pull Request

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Connect

* **Blog:** [chapainaashish.hashnode.dev](https://chapainaashish.hashnode.dev)
* **LinkedIn:** [linkedin.com/in/chapainaashish](https://linkedin.com/in/chapainaashish)
* **GitHub:** [@chapainaashish](https://github.com/chapainaashish)
* **Website:** [chapainaashish.com.np](https://chapainaashish.com.np)

---

Built with ❤️ by [Aashish Chapain](https://github.com/chapainaashish)


