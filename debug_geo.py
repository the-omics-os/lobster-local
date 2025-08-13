from lobster.tools.pubmed_service import PubMedService

from core.data_manager import DataManager
from dotenv import load_dotenv
import os

load_dotenv()

data_manager = DataManager()

pubmed_service = PubMedService(parse=None, data_manager=data_manager, api_key=os.environ.get('NCBI_API_KEY'))

pubmed_service.find_geo_from_doi('10.1186/s13073-023-01164-9')