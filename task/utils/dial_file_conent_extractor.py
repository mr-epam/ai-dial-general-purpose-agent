import io
from pathlib import Path

import pdfplumber
import pandas as pd
from aidial_client import Dial
from bs4 import BeautifulSoup


class DialFileContentExtractor:

    def __init__(self, endpoint: str, api_key: str):
        #TODO:
        # Set Dial client with endpoint as base_url and api_key
        self.client = Dial(base_url=endpoint, api_key=api_key)

    def extract_text(self, file_url: str) -> str:
        #TODO:
        # 1. Download with Dial client file by `file_url` (files -> download)
        response = self.client.files.download(url=file_url)
        # 2. Get downloaded file name and content
        filename = response.filename # response._response.headers.get('Content-Disposition', '').split('filename=')[-1].strip('"')
        file_content = response.get_content() # response._response.content
        # 3. Get file extension, use for this `Path(filename).suffix.lower()`
        file_extension = Path(filename).suffix.lower()
        # 4. Call `__extract_text` and return its result
        return self.__extract_text(file_content=file_content, file_extension=file_extension, filename=filename)

    def __extract_text(self, file_content: bytes, file_extension: str, filename: str) -> str:
        """Extract text content based on file type."""
        #TODO:
        # Wrap in `try-except` block:
        # try:
        #   1. if `file_extension` is '.txt' then return `file_content.decode('utf-8', errors='ignore')`
        #   2. if `file_extension` is '.pdf' then:
        #       - load it with `io.BytesIO(file_content)`
        #       - with pdfplumber.open PDF files bites
        #       - iterate through created pages adn create array with extracted page text
        #       - return it joined with `\n`
        #   3. if `file_extension` is '.csv' then:
        #       - decode `file_content` with encoding 'utf-8' and errors='ignore'
        #       - create csv buffer from `io.StringIO(decoded_text_content)`
        #       - read csv with pandas (pd) as dataframe
        #       - return dataframe to markdown (index=False)
        #   4. if `file_extension` is in ['.html', '.htm'] then:
        #       - decode `file_content` with encoding 'utf-8' and errors='ignore'
        #       - create BeautifulSoup with decoded html content, features set as 'html.parser' as `soup`
        #       - remove script and style elements: iterate through `soup(["script", "style"])` and `decompose` those scripts
        #       - return `soup.get_text(separator='\n', strip=True)`
        #   5. otherwise return it as decoded `file_content` with encoding 'utf-8' and errors='ignore'
        # except:
        #   print an error and return empty string
        try:
            if file_extension == '.txt':
                return file_content.decode('utf-8', errors='ignore')
            elif file_extension == '.pdf':
                with io.BytesIO(file_content) as pdf_file:
                    with pdfplumber.open(pdf_file) as pdf:
                        pages_text = [page.extract_text() or '' for page in pdf.pages]
                return '\n'.join(pages_text)
            elif file_extension == '.csv':
                decoded_text_content = file_content.decode('utf-8', errors='ignore')
                csv_buffer = io.StringIO(decoded_text_content)
                df = pd.read_csv(csv_buffer)
                return df.to_markdown(index=False)
            elif file_extension in ['.html', '.htm']:
                decoded_html_content = file_content.decode('utf-8', errors='ignore')
                soup = BeautifulSoup(decoded_html_content, 'html.parser')
                for script_or_style in soup(['script', 'style']):
                    script_or_style.decompose()
                return soup.get_text(separator='\n', strip=True)
            else:
                return file_content.decode('utf-8', errors='ignore')
            
        except Exception as e:
            print(f"Error extracting content from {filename}: {e}")
            return ""
