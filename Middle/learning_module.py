from retrieval_module import Memory
from typing import List
from groq_interface import Groq_Agent
import re

class Learner:
    def __init__(self, memory_module: Memory, llm_interface:Groq_Agent):
        self.mem = memory_module
        self.llm = llm_interface

    def _add_new_top_level_domains(self, domain_names:List[tuple]):
        for field_name, general_string in domain_names:
            self.mem.add_top_level_field(field_name, general_string)

    def learn_from_query(self, query, retrieved_context):
        # query: user's last interaction + context summary till now
        # retrieved context: Stuff we already know about this query
        learning_prompt = f"""
            The user's last query and the conversation so far: 
            {query}

            Top level knowledge domains:
            {" ".join(self.mem.top_level_fields)}

            Retrieved context, stuff we ALREADY KNOW (only for reference, not to be learned from):
            {retrieved_context}
            
            1. Figure out what, if any, information is contained in the user's query that is not there in the context
                in other words, we want to learn <info in query> - <info retrieved>
            2. Learn only significant facts
            Also, figure out if we need NEW top level domains i.e. NOT in those listed above
            Your entire output should be in this format, and contain nothing else
            ""
            <new>
            ...(new, single-word, top level domain) | (general info. about this domain)
            e.g. "health | Ishaan reports being in good health"
            <new>
            \n\n
            <fact>
            ...(New facts and which top level domain they come under)
            (e.g. "<existing(/newly added)_top_level_domain> |His sister's name is <>")
            <fact>
            ""
            Do not return <new>...<new> and <fact>...<fact> tags, if they're empty
            BE VERY, VERY CONCISE AND PRECISE. No text outside tags. 
            Store the facts in a personal tone
        """
        (_, resp, _) = self.llm.make_query(learning_prompt, temp=0.5)
        # Add necessary new top level domains
        new_domain_matches_rough = re.search(r"<new>(.*?)<new>", resp, re.DOTALL)
        if new_domain_matches_rough is not None:
            new_domain_matches = new_domain_matches_rough.group(1).strip()
            new_domains_to_add = new_domain_matches.split("\n")
            clean = []
            for i in new_domains_to_add:
                if "|" in i:
                    domain_name, g_string = i.split("|")
                    clean.append((domain_name, g_string))
                else:
                    clean.append((domain_name, ""))
            self._add_new_top_level_domains(clean)
        # Okay, now add/alter fields in existing top level domains
        fact_matches = re.search(r"<fact>(.*?)<fact>", resp, re.DOTALL)
        if fact_matches is None:
            fact_matches = re.search(r"<fact>(.*?)<\fact>", resp, re.DOTALL)
        if fact_matches is None:
            # no facts to add, skip
            return
        fact_matches = fact_matches.group(1).strip()
        facts_to_add = fact_matches.split("\n")
        domain_fact = {}
        for fact in facts_to_add:
            domain, factoid = fact.split("|")
            if domain not in domain_fact:
                domain_fact[domain.strip()] = []
            domain_fact[domain.strip()].append(factoid)
        # now, go top level domain by domain, and add/alter it 
        for top_level_domain, new_facts_list in domain_fact.items():
            if top_level_domain not in self.mem.field_data:
                continue
            existing_fields = list(self.mem.data[top_level_domain].keys())
            prompt = f"""
            Here are the existing fields of this domain:
            {", ".join(existing_fields)}
            The new facts we want to add:
            {"\n".join(new_facts_list)}
            Please, for each change you're SURE you want to make to this domain's data, return an item:
            <Add/Alter> |<single-word new/existing field_name>| "<fact_string>"
            Return all output within <ans>.. <ans> tags, NO EXTRA TEXT
            e.g. 
            <ans>
            Add | sister | "Aadya, 5 years younger",
            Alter | motivation | "After experiencing travel and parties, earning money has become a bigger motivation for Ishaan"
            <ans>
            This example will associate "sister" key with "Aadya, 5 years younger"
            Also, don't assume any context, like write "that day" for "today"
            """
            (_, resp, _) = self.llm.make_query(prompt, temp=0.1)
            changes = re.search(r"<ans>(.*?)<ans>", resp, re.DOTALL).group(1).strip().split("\n") # List of changes
            for change in changes:
                x, field, fact_string = list(map(lambda x:x.strip(), change.split("|")))
                self.mem.change_subfield_and_fact(top_level_domain, field, fact_string, (x.strip().lower()=="add"))

        # Return the last learned fact(to keep track of the next convo)
        return f"Last learned fact: {field}: {fact_string}"
            
