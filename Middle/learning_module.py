from retrieval_module import Memory
from typing import List
from groq_interface import Groq_Agent
import re

class Learner:
    def __init__(self, memory_module: Memory, llm_interface:Groq_Agent):
        self.mem = memory_module
        self.llm = llm_interface

    def _add_new_top_level_domains(self, domain_names:List[str]):
        for field_name in domain_names:
            self.mem.add_top_level_field(field_name)

    def learn_from_query(self, query, retrieved_context):
        # query: user's last interaction + context summary till now
        # retrieved context: Stuff we already know about this query
        learning_prompt = f"""
            The user's last query and the conversation so far: 
            {query}

            Retrieved context, stuff we already know:
            {retrieved_context}

            Top level knowledge domains:
            {" ".join(self.mem.top_level_fields)}

            Figure out what, if any, NEW information is contained in the user's query.
            Err on the side of including MORE.
            Also, figure out if we need new top level domains.
            Then, output this information, piece by piece with top level domains
            in this format:
            <new>
            ...(new, single-word, top level domains we require...line separated)
            <new>
            <fact>
            ...(New facts and which top level domain they come under)
            (e.g. "<existing(/newly added)_top_level_domain> |His sister's name is <>")
            <fact>
        """
        (_, resp, _) = self.llm.make_query(learning_prompt)
        # Add necessary new top level domains
        new_domain_matches = re.search(r"<new>(.*?)<new>", resp, re.DOTALL).group(1).strip()
        new_domains_to_add = new_domain_matches.split("\n")
        self._add_new_top_level_domains(new_domains_to_add)
        # Okay, now add/alter fields in existing top level domains
        fact_matches = re.search(r"<fact>(.*?)<fact>", resp, re.DOTALL).group(1).strip()
        facts_to_add = fact_matches.split("\n")
        domain_fact = {}
        for fact in facts_to_add:
            domain, factoid = fact.split("|")
            if domain not in domain_fact:
                domain_fact[domain.strip()] = []
            domain_fact[domain.strip()].append(factoid)
        # now, go top level domain by domain, and add/alter it 
        for top_level_domain, new_facts_list in domain_fact.items():
            existing_fields = list(self.mem.data[top_level_domain].keys())
            prompt = f"""
            Here are the existing fields of this domain:
            {", ".join(existing_fields)}
            The new facts we want to add:
            {"\n".join(new_facts_list)}
            Please, for each change you want to make to this domain's data, return an item:
            <Add/Alter> |<single-word new/existing field_name>| "<fact_string>"
            Return all output within <ans>.. <ans> tags, NO EXTRA TEXT
            e.g. 
            <ans>
            Add | sister | "Aadya, 5 years younger" 
            <ans>
            This example will associate "sister" key with "Aadya, 5 years younger"
            """
            (_, resp, _) = self.llm.make_query(prompt)
            changes = re.search(r"<ans>(.*?)<ans>", resp, re.DOTALL).group(1).strip().split("\n") # List of changes
            for change in changes:
                x, field, fact_string = list(map(lambda x:x.strip(), change.split("|")))
                self.mem.change_subfield_and_fact(top_level_domain, field, fact_string, (x.strip().lower()=="add"))
            
