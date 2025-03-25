from typing import List
import json
import datetime
from groq_interface import Groq_Agent

class Memory:
    def init_temporary(self, basic_info, events):
        # Update date, time, current age
        # basic_info is a python dict
        dob_str = basic_info["DOB"]
        dob = datetime.strptime(dob_str, "%d/%m/%Y")

        today = datetime.today()
        date_str1 = today.strftime("%d/%m/%Y")  # "25/03/2025"
        # Compute age in years
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        
        events_today = []
        for date, description in events:
            if date == date_str1:
                events_today.append(description)

        return (date, age, events_today)


    def __init__(self, llm_interface: Groq_Agent, path_to_permanent_data="KB/permanent.json") -> None:
        permanent_info = self.get_info(path_to_permanent_data)
        self.data = permanent_info['data']
        self.basic_info = permanent_info['basic_info']
        self.field_data = permanent_info['fields_info']
        self.top_level_fields = list(self.field_data.keys())
        self.init_temporary(self.basic_info, 
                            self.data["events"]) # Initialise temporary data file
        self.llm = llm_interface

    def get_info(self, path_to_permanent_data):
        # Read the permanent.json file and store it in memory
        with open(path_to_permanent_data, "r") as file:
            return json.load(file)

    def get_basic_info(self, field):
        # Return basic information
        if field in self.permanent_info["basic_info"]:
            return self.permanent_info["basic_info"][field]
        return None
    
    def get_date(self):
        today = datetime.today()
        return today.strftime("%d/%m/%Y")  # "25/03/2025"

    # Retrieving data
    def _generate_keys(self, contextualized_query)->List[str]:
        # Which top level fields to access
        # Send query+context, and list of top level fields
        """
        LLM output should be in the format 
        <keys>
        top_level,search_terms...
        * as many as it likes
        <\keys>
        """
        return [[],[]]

    def retrieve(self, query):
        # I'll return a string with all the data
        recalled_data = []
        hierarchical_keys = self._generate_keys(query)
        # The first element of each sub-array is my top-level keys
        for key_obj in hierarchical_keys:
            top_level_key = key_obj[0]
            if top_level_key in self.field_data:
                data_obj = self.data[top_level_key]
                recalled_data.append(data_obj["general"]) # Essentials
                if len(key_obj) > 1:
                    # Actually has some search terms
                    flag = 0
                    for search_term in key_obj[1:]:
                        if search_term in data_obj:
                            recalled_data.append(data_obj[search_term])
                            flag = 1
                    if not flag:
                        recalled_data.append(data_obj["misc."])
        return "\n".join(recalled_data)