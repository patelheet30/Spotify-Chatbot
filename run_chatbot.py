import logging
import os

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_environment():
    load_dotenv()

    required_envs = ["SPOTIPY_CLIENT_ID", "SPOTIPY_CLIENT_SECRET", "DATABASE_URL"]
    missing_envs = [env for env in required_envs if not os.getenv(env)]

    if missing_envs:
        print("Error: The following environment variables are missing:")
        for env in missing_envs:
            print(f"  - {env}")
        print("\nPlease create a .env file in the project root with the following:")
        print("""
SPOTIPY_CLIENT_ID=your_spotify_client_id
SPOTIPY_CLIENT_SECRET=your_spotify_client_secret
DATABASE_URL=sqlite:///./spotify_chatbot.db
        """)
        return False

    dirs_to_check = ["static", "static/uploads", "output"]
    for directory in dirs_to_check:
        if not os.path.exists(directory):
            logger.info(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)

    index_html_path = "static/index.html"
    if not os.path.exists(index_html_path):
        with open(index_html_path, "w") as f:
            f.write("<!-- This file will be overwritten by the application -->")

    return True


def run_web_interface():
    try:
        import uvicorn

        logger.info("Starting web interface on http://localhost:8000")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        logger.error(
            "Uvicorn is not installed. Please install it with: pip install uvicorn"
        )
        return False
    except Exception as e:
        logger.error(f"Error starting web interface: {e}")
        return False

    return True


def run_terminal_interface():
    try:
        import asyncio

        from app.chatbot.test.test_handler import test_chatbot

        logger.info("Starting Terminal interface")
        asyncio.run(test_chatbot())
    except ImportError as e:
        logger.error(f"Error importing test_handler: {e}")
        return False
    except Exception as e:
        logger.error(f"Error running Terminal interface: {e}")
        return False

    return True


def main():
    print("=" * 50)
    print("  Spotify Music Chatbot")
    print("=" * 50)

    if not setup_environment():
        return

    while True:
        print("\nSelect an interface:")
        print("1. Web Interface")
        print("2. Terminal Interface")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == "1":
            run_web_interface()
            break
        elif choice == "2":
            run_terminal_interface()
            break
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
