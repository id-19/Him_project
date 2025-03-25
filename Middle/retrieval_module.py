from typing import List
import json
from datetime import datetime
import re
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

        return (date_str1, age, events_today)


    def __init__(self, llm_interface: Groq_Agent, path_to_permanent_data="KB/permanent.json") -> None:
        permanent_info = self.get_info(path_to_permanent_data)
        self.data = permanent_info['data']
        self.basic_info = permanent_info['basic_info']
        self.field_data = permanent_info['fields_info']
        self.top_level_fields = list(self.field_data.keys())
        # self.init_temporary(self.basic_info, 
                            # self.data["events"]) # Initialise temporary data file
        self.llm = llm_interface

    def get_info(self, path_to_permanent_data):
        # Read the permanent.json file and store it in memory
        with open(path_to_permanent_data, "r") as file:
            return json.load(file)

    def get_basic_info(self, field):
        # Return basic information
        return self.basic_info.get(field, None)
    
    def get_date(self):
        today = datetime.today()
        return today.strftime("%d/%m/%Y")  # "25/03/2025"

    # Retrieving data
    def _generate_keys(self, contextualized_query)->List[str]:
        # Generate keys to search through knowledge base
        """
        LLM output should be in the format 
        <keys>
        top_level,search_terms...
        * as many as it likes
        <\keys>
        """
        # Send query+context, and list of top level fields
        prompt = f"""
        Here is the context + query:{contextualized_query}
        Here are the top level fields in my knowledge base:{' '.join(self.top_level_fields)}
        I want you to generate search keys for my knowledge base,
        and return them in this format:
        <keys>
        top_level_field
        ...
        <\keys>
        1. Do not generate any text outside the <keys> 
        2. Only generate top level fields
        """
        (_,resp,_) = self.llm.make_query(prompt)

        # Extract text inside <keys>...</keys> using regex
        match = re.search(r"<keys>(.*?)</keys>", resp, re.DOTALL)
        
        if not match:
            return []  # Return an empty list if no valid keys were found

        keys_content = match.group(1).strip()
        top_level_keys = keys_content.split("\n")
        rough_keys = []
        for top_level_key in top_level_keys:
            rough_keys.append(f"{top_level_key} | {list(self.data[top_level_key].keys())}")

        final_prompt = f"""
        Here is the context + query:{contextualized_query}
        Now, I've chosen some fields from my knowledge base. I want you to select the subfields that seem relevant to you.
        Here is a list of top_level_field | <chosen keys in top level field>:
        {"\n".join(rough_keys)}
        Return your output in the format:
        <keys>
        <top_level_field> | <chosen search term1>, <chosen term2> ....
        ...
        <\keys>
        1. Err on the side of slightly more
        2. DO NOT OUTPUT ANY TEXT OUTSIDE <keys>...</keys>
        """
        (_, resp,_) = self.llm.make_query(final_prompt)
        # Extract text inside <keys>...</keys> using regex
        match1 = re.search(r"<keys>(.*?)</keys>", resp, re.DOTALL)
        
        if not match1:
            return []  # Return an empty list if no valid keys were found

        keys_content = match1.group(1).strip()
        key_list = []
        
        for line in keys_content.split("\n"):
            line = line.strip()
            if "|" in line:
                field, terms = line.split("|", 1)
                key_list.append([field.strip()] + [term.strip() for term in terms.split(",") if term.strip()])
        
        return key_list

    def retrieve(self, contextualized_query):
        # I'll return a string with all the data

        # def search_misc(search_term, misc_data):
        #     # "misc.": [...set of strings] // for all the stuff that doesn't necessarily match with a key
        #     res = []
        #     for data_str in misc_data:
        #         if search_term in data_str:
        #             res.append(data_str)
        #     return res
            
        recalled_data = []
        hierarchical_keys = self._generate_keys(contextualized_query)
        # The first element of each sub-array is my top-level keys
        for key_obj in hierarchical_keys:
            top_level_key = key_obj[0]
            if top_level_key in self.field_data:
                data_obj = self.data[top_level_key]
                recalled_data.append(data_obj["general"]) # Essentials

                if len(key_obj) > 1:
                    # Actually has some search terms
                    search_terms = key_obj[1:]
                    recalled_data.extend(f"{term}:{data_obj[term]}" for term in search_terms if term in data_obj)
                    # misc_extension = [search_misc(term, data_obj["misc."]) for term in search_terms]
                    # if len(misc_extension) > 0:
                    #     recalled_data.extend(misc_extension)
                    
        return "\n".join(recalled_data)