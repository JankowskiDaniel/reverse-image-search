from pydantic import BaseModel
from typing import List, Dict

class Status(BaseModel):
    connection: bool