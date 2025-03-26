import concurrent.futures
import os
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

################################################################################
# Setup: Directories, shared variables, etc.
################################################################################

base_dir = "album_covers_data"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Example range of years (adjust as needed)
years = list(range(1980, 2001))

# Create the directory structure
for year in years:
    os.makedirs(os.path.join(train_dir, str(year)), exist_ok=True)
    os.makedirs(os.path.join(test_dir, str(year)), exist_ok=True)

print("Directory structure created successfully!")

################################################################################
# Utility Functions
################################################################################


def sanitize_filename(name: str) -> str:
    """
    Replaces problematic characters in a filename with underscores.
    """
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def download_image(url: str, save_path: str) -> bool:
    """
    Downloads an image from the specified URL and saves it to `save_path`.
    Returns True if successful, False otherwise.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        response = requests.get(url, stream=True, timeout=10, headers=headers)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            print(f"Failed to download {url}, status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


################################################################################
# Extracting Album Links - New Format (e.g., 2005+)
################################################################################


def extract_album_links_from_page(url: str) -> list[dict]:
    """
    Extracts album links from a Wikipedia page using the 'newer' format:
    e.g. https://en.wikipedia.org/wiki/List_of_2005_albums
    Returns a list of dicts with keys: "title", "url", "artist".
    """
    album_links = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"Failed to fetch {url}, status code: {response.status_code}")
            return album_links

        soup = BeautifulSoup(response.text, "html.parser")
        tables = soup.find_all("table", class_="wikitable")

        for table in tables:
            rows = table.find_all("tr")
            # Skip header row
            for row in rows[1:]:
                cells = row.find_all("td")
                # Typically we expect at least 2 columns: Artist / Album
                if len(cells) >= 2:
                    # Second cell often has album name
                    links = cells[1].find_all("a")
                    for link in links:
                        href = link.get("href")
                        if (
                            href
                            and "/wiki/" in href
                            and not any(
                                x in href
                                for x in [
                                    "File:",
                                    "redlink=1",
                                    "wikipedia",
                                    "Category:",
                                ]
                            )
                        ):
                            album_links.append(
                                {
                                    "title": link.text.strip(),
                                    "url": f"https://en.wikipedia.org{href}",
                                    "artist": cells[0].text.strip(),
                                }
                            )
    except Exception as e:
        print(f"Error fetching album links from {url}: {e}")

    return album_links


################################################################################
# Extracting Album Links - Old Format (e.g., 2004_in_music#Albums_released)
################################################################################


def extract_album_links_from_old_page(url: str) -> list[dict]:
    """
    Extracts album links from the older Wikipedia page format, e.g.:
    https://en.wikipedia.org/wiki/2004_in_music#Albums_released

    Returns a list of dicts with keys: "title", "url", "artist".
    """
    album_links = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"Failed to fetch {url}, status code: {response.status_code}")
            return album_links

        soup = BeautifulSoup(response.text, "html.parser")

        # In older pages, the album tables are often found under the heading
        # "Albums released" in multiple "wikitable" tables (Jan–Mar, Apr–Jun, etc.)
        tables = soup.find_all("table", class_="wikitable")

        for table in tables:
            rows = table.find_all("tr")
            # Skip header row
            for row in rows[1:]:
                cells = row.find_all("td")
                # Typically: [#, Album, Artist, Notes]
                if len(cells) >= 3:
                    album_cell = cells[0]
                    artist_cell = cells[1]

                    # Find all links in the album cell
                    links = album_cell.find_all("a")
                    for link in links:
                        href = link.get("href")
                        if (
                            href
                            and "/wiki/" in href
                            and not any(
                                x in href
                                for x in [
                                    "File:",
                                    "redlink=1",
                                    "wikipedia",
                                    "Category:",
                                ]
                            )
                        ):
                            album_links.append(
                                {
                                    "title": link.text.strip(),
                                    "url": f"https://en.wikipedia.org{href}",
                                    "artist": artist_cell.text.strip(),
                                }
                            )

    except Exception as e:
        print(f"Error fetching album links from {url}: {e}")

    return album_links


################################################################################
# Common Album Processing (for concurrency)
################################################################################


def process_album(album_link: dict, i: int, total: int, year: int) -> dict | None:
    """
    Fetches a single album's Wikipedia page, parses the infobox to find an image,
    downloads the image, and returns metadata if successful.
    """
    try:
        print(f"Processing {i + 1}/{total}: {album_link['title']}")
        album_response = requests.get(album_link["url"], timeout=15)
        if album_response.status_code != 200:
            print(f"Failed to fetch {album_link['url']}")
            return None

        album_soup = BeautifulSoup(album_response.text, "html.parser")
        infobox = album_soup.find("table", class_="infobox")
        if not infobox:
            print(f"No infobox found for {album_link['title']}")
            return None

        img_element = infobox.find("img")
        if not img_element:
            print(f"No image found for {album_link['title']}")
            return None

        img_src = img_element.get("src")
        if not img_src:
            print(f"No image source for {album_link['title']}")
            return None

        # Ensure full URL
        if img_src.startswith("//"):
            img_url = f"https:{img_src}"
        else:
            img_url = img_src

        # Create a filename
        safe_artist = sanitize_filename(album_link["artist"])
        safe_title = sanitize_filename(album_link["title"])
        filename = f"{year}_{safe_artist}_{safe_title}.jpg"

        # 80/20 split: every 5th album goes to test
        is_train = i % 5 != 0
        destination = "train" if is_train else "test"
        save_path = os.path.join(base_dir, destination, str(year), filename)

        if download_image(img_url, save_path):
            print(f"Downloaded: {album_link['artist']} - {album_link['title']}")
            return {
                "year": year,
                "artist": album_link["artist"],
                "album": album_link["title"],
                "filename": filename,
                "source_url": album_link["url"],
                "image_url": img_url,
                "dataset": destination,
            }
    except Exception as e:
        print(f"Error processing album {album_link['title']}: {e}")

    return None


################################################################################
# Gathering Albums - Newer Format
################################################################################


def get_wikipedia_albums(year: int) -> list[dict]:
    """
    Gathers all albums for a given year using the newer format, e.g.:
    https://en.wikipedia.org/wiki/List_of_2005_albums
    Then downloads the album covers concurrently.
    """
    print(f"Fetching albums from {year} (new format)...")
    album_data = []
    album_links = []
    urls_to_try = [
        f"https://en.wikipedia.org/wiki/List_of_{year}_albums",
        f"https://en.wikipedia.org/wiki/List_of_{year}_albums_(January%E2%80%93June)",
        f"https://en.wikipedia.org/wiki/List_of_{year}_albums_(July%E2%80%93December)",
    ]

    found_pages = False
    for url in urls_to_try:
        print(f"Trying URL: {url}")
        links = extract_album_links_from_page(url)
        if links:
            found_pages = True
            print(f"Found {len(links)} album links from {url}")
            album_links.extend(links)
            break
        time.sleep(1)

    if not found_pages:
        print(f"No album lists found for {year} using the newer format.")
        return album_data

    print(f"Total album links found for {year}: {len(album_links)}")
    unique_albums = []
    unique_urls = set()

    # Remove duplicates
    for album in album_links:
        if album["url"] not in unique_urls:
            unique_albums.append(album)
            unique_urls.add(album["url"])
            # also skip duplicates by same title or artist if desired
    album_links = unique_albums
    print(f"After removing duplicates: {len(album_links)} unique albums")

    # Download images concurrently
    total = len(album_links)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(process_album, album, i, total, year): album
            for i, album in enumerate(album_links)
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                album_data.append(result)

    return album_data


################################################################################
# Gathering Albums - Older Format
################################################################################


def get_wikipedia_albums_old_format(year: int) -> list[dict]:
    """
    Gathers all albums for a given year using the older format, e.g.:
    https://en.wikipedia.org/wiki/2004_in_music#Albums_released
    Then downloads the album covers concurrently.
    """
    print(f"Fetching albums from {year} (old format)...")
    album_data = []

    # Typically the old format is found at {year}_in_music#Albums_released
    url = f"https://en.wikipedia.org/wiki/{year}_in_music#Albums_released"
    print(f"Trying URL: {url}")
    links = extract_album_links_from_old_page(url)
    print(f"Found {len(links)} album links from old-format page")

    if not links:
        print(f"No album links found for {year} using old format.")
        return album_data

    # Remove duplicates
    unique_albums = []
    unique_urls = set()
    for album in links:
        if album["url"] not in unique_urls:
            unique_albums.append(album)
            unique_urls.add(album["url"])
    album_links = unique_albums
    print(f"After removing duplicates: {len(album_links)} unique albums")

    # Download images concurrently
    total = len(album_links)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(process_album, album, i, total, year): album
            for i, album in enumerate(album_links)
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                album_data.append(result)

    return album_data


################################################################################
# Main "Generate Dataset" Function
################################################################################


def generate_dataset(years: list[int]):
    """
    High-level function to gather album data for each year, saving images and
    metadata. For older years (<= 2004), uses the old format function;
    for newer years, uses the new format function.
    """
    all_album_data = []

    for year in years:
        # Decide which approach to use based on the year
        if year <= 2004:
            year_data = get_wikipedia_albums_old_format(year)
        else:
            year_data = get_wikipedia_albums(year)

        all_album_data.extend(year_data)

        # Save progress after each year
        df = pd.DataFrame(all_album_data)
        df.to_csv("album_dataset_info.csv", index=False)

        print(f"Completed {year}, total albums collected this year: {len(year_data)}")
        print(f"Running total albums: {len(all_album_data)}")

        # Be polite to Wikipedia (adjust as needed)
        time.sleep(5)

    print(f"\nData collection complete! Total albums collected: {len(all_album_data)}")
    print("The album metadata is saved in 'album_dataset_info.csv'.")
    print(f"The images are organized in the '{base_dir}' directory.")


################################################################################
# Optional: Checking Album Counts Without Downloading
################################################################################


def check_album_counts(years: list[int]):
    """
    Checks the total count of albums for each year without downloading images.
    For older years, it uses the old page extraction;
    for newer years, it uses the new page extraction.
    """
    print("Checking album counts for each year (no downloads)...")
    album_counts = {}

    for year in years:
        if year <= 2004:
            # Old format
            url = f"https://en.wikipedia.org/wiki/{year}_in_music#Albums_released"
            links = extract_album_links_from_old_page(url)
        else:
            # New format
            # We'll just try the first relevant URL in the list_of_year_albums approach
            url = f"https://en.wikipedia.org/wiki/List_of_{year}_albums"
            links = extract_album_links_from_page(url)

        album_counts[year] = len(links)
        print(f"{year}: found {len(links)} album links.")

    print("\nAlbum counts by year:")
    for year, count in album_counts.items():
        print(f"{year}: {count} albums")

    return album_counts


################################################################################
# Entry Point
################################################################################

if __name__ == "__main__":
    # Example usage: generate the dataset for 2000-2008
    generate_dataset(years)

    # Or, just check counts:
    # check_album_counts(years)
